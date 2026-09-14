"""
Grounding DINO Adapter for Open-Vocabulary Object Detection.
Provides a standardized interface for local, air-gapped Grounding DINO inference
conforming to the project's detection module contract.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src import config
from src.detection.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class GroundingDINOAdapter:
    """
    Air-gapped adapter for Grounding DINO zero-shot object detection models.
    Handles offline model initialization, prompt formatting, inference, and
    output standardization.
    """

    def __init__(self, model_directory_name: str = "groundingdino"):
        """
        Initializes Grounding DINO from local storage.

        Args:
            model_directory_name: Folder inside models/ containing the weights.
        """
        self.model_path = config.MODELS_DIR / model_directory_name
        self.prompt_manager = PromptManager()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Air-gapped Grounding DINO weights not found at {self.model_path}. "
                "Ensure local weights exist in the models directory."
            )

        logger.info(f"Loading local Grounding DINO processor from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        logger.info(
            f"Loading local Grounding DINO model to {config.DEVICE} in {config.TORCH_DTYPE}"
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=config.TORCH_DTYPE,
        ).to(config.DEVICE)

        self.model.eval()

    def ingest_image(
        self, image_data: Union[Path, np.ndarray, Image.Image]
    ) -> Image.Image:
        """
        Standardizes input raster tile data into a 3-channel RGB PIL Image.
        """
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        if isinstance(image_data, Path):
            if not image_data.exists():
                raise FileNotFoundError(f"Image tile not found at {image_data}")
            return Image.open(image_data).convert("RGB")

        if isinstance(image_data, np.ndarray):
            if image_data.ndim == 3 and image_data.shape[0] == 3:
                transposed_array = np.transpose(image_data, (1, 2, 0))
            else:
                transposed_array = image_data

            if transposed_array.dtype != np.uint8:
                transposed_array = transposed_array.astype(np.uint8)

            return Image.fromarray(transposed_array, mode="RGB")

        raise TypeError(f"Unsupported image input type: {type(image_data)}")

    def predict(
        self,
        image_data: Union[Path, np.ndarray, Image.Image],
        targets: Union[str, List[str]],
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes zero-shot object detection on a single raster tile.

        Args:
            image_data: Input tile (Path, NumPy array, or PIL Image).
            targets: Target prompt string or list of target keys (e.g., 'tents', 'trenchlines').
            box_threshold: Minimum bounding box confidence score threshold.
            text_threshold: Minimum text-token alignment threshold.

        Returns:
            List of dictionaries: [{"bbox": [xmin, ymin, xmax, ymax], "label": str, "confidence": float}]
        """
        image = self.ingest_image(image_data)

        # Retrieve target-specific threshold defaults if not explicitly overridden
        primary_target_key = (
            targets[0] if isinstance(targets, list) and targets else str(targets)
        )
        recommended_thresholds = self.prompt_manager.get_thresholds(
            primary_target_key,
            override_box=box_threshold,
            override_text=text_threshold,
        )

        b_thresh = recommended_thresholds["box_threshold"]
        t_thresh = recommended_thresholds["text_threshold"]

        # Format prompt syntax (period-separated terms required by Grounding DINO)
        prompt_text = self.prompt_manager.format_for_grounding_dino(
            targets, include_synonyms=True
        )

        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(config.TORCH_DTYPE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs to absolute tile pixel coordinates
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            box_threshold=b_thresh,
            text_threshold=t_thresh,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        standardized_results: List[Dict[str, Any]] = []

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            standardized_results.append(
                {
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],  # [xmin, ymin, xmax, ymax]
                    "label": str(label),
                    "confidence": float(score),
                }
            )

        return standardized_results