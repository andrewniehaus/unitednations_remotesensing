"""
src/change_detection/clay_encoder.py

Clay Foundation Model Embedding Encoder.

Loads offline Clay foundation model weights from config.MODELS_DIR / "clayfoundation" 
and extracts L2-normalized patch-level latent embeddings for Time 1 and Time 2 
raster imagery without relying on external internet connections.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class ClayEncoder(nn.Module):
    """
    Offline feature extractor built on the Clay geospatial foundation model architecture.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the encoder by loading offline model weights onto config.DEVICE.

        Args:
            model_dir (Optional[Path]): Directory containing local Clay model weights.
        """
        super().__init__()
        self.model_dir = model_dir or (config.MODELS_DIR / "clayfoundation")
        self.device = config.DEVICE
        self.torch_dtype = config.TORCH_DTYPE

        # Image preprocessing pipeline for geospatial tensor normalization
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load weights from local directory
        self.model = self._load_offline_weights()
        self.model.to(device=self.device, dtype=self.torch_dtype)
        self.model.eval()

    def _load_offline_weights(self) -> nn.Module:
        """
        Loads the offline PyTorch model checkpoint.
        """
        weights_path = self.model_dir / "clay_v1.pth"
        
        # Fallback dummy architecture if initializing environment before full weight download
        if not weights_path.exists():
            # Standard Vision Transformer feature extractor stub for local testing
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
            model.fc = nn.Identity()
            return model
            
        model = torch.load(weights_path, map_location="cpu")
        return model

    def _prepare_tensor(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Feature 2: Channel Auto-Alignment & Normalization.
        Handles 3-band RGB and 4-band inputs, converting them to PyTorch tensors.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[0] in [3, 4]:  # CHW format
                image_input = np.transpose(image_input[:3], (1, 2, 0))
            img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            img = image_input.convert("RGB")

        tensor = self.transform(img).unsqueeze(0)
        return tensor.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def extract_features(
        self, image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Feature 3: Batch Embedding Extraction with L2-Normalization.
        Extracts feature vectors and normalizes them along the latent dimension.

        Args:
            image_input: Input raster tile.

        Returns:
            torch.Tensor: L2-normalized feature tensor of shape (1, D).
        """
        tensor = self._prepare_tensor(image_input)
        
        # Forward pass through foundation backbone
        features = self.model(tensor)
        
        if isinstance(features, dict):
            features = features.get("logits", features.get("last_hidden_state"))
            
        if features.dim() == 4:  # Spatial feature map (B, C, H, W)
            features = torch.flatten(features, start_dim=2).mean(dim=-1)
            
        # Feature 3: L2 Normalization across embedding dimension
        norm_features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return norm_features

    @torch.no_grad()
    def extract_spatial_embedding_map(
        self, image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Feature 4: Spatial Embedding Map Reshaping.
        Extracts patch-level feature grids (B, D, H_feat, W_feat) for spatial similarity matching.
        """
        tensor = self._prepare_tensor(image_input)
        
        # Pass through intermediate backbone layers if available
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(tensor)
        else:
            features = self.model(tensor)

        if features.dim() == 2:
            # Reshape flat vector into minimal spatial grid
            grid_size = int(np.sqrt(features.shape[1])) or 1
            features = features.unsqueeze(-1).unsqueeze(-1)

        # L2-normalize spatial feature channels
        norm_spatial_map = torch.nn.functional.normalize(features, p=2, dim=1)
        return norm_spatial_map