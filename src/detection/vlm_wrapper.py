import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src import config

logger = logging.getLogger(__name__)


class LocalVLMWrapper:
    """
    Air-gapped Vision-Language Model wrapper for offline object detection.
    Handles hardware allocation, local weight loading, and array-to-tensor ingestion.
    """

    def __init__(self, model_directory_name: str):
        """
        Step 1: Initialization from Local Storage
        Loads weights strictly from the local models directory and maps them to 
        the predefined hardware device and precision type.
        
        Args:
            model_directory_name: The folder name inside data/models/ containing the weights 
                                  (e.g., 'florence-2-large').
        """
        self.model_path = config.MODELS_DIR / model_directory_name
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Air-gapped model weights not found at {self.model_path}. "
                "Ensure weights are downloaded and placed in the models directory."
            )

        logger.info(f"Loading local processor from {self.model_path}")
        # local_files_only=True strictly enforces the air-gapped constraint
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        logger.info(f"Loading local model to {config.DEVICE} in {config.TORCH_DTYPE}")
        # Initialize model with memory-efficient dtype and push to specified GPU device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=config.TORCH_DTYPE,
            trust_remote_code=True
        ).to(config.DEVICE)

        # Lock model in evaluation mode to prevent gradient tracking (saves VRAM)
        self.model.eval()

    def ingest_image(self, image_data: Union[Path, np.ndarray]) -> Image.Image:
        """
        Step 2: Data Ingestion
        Accepts normalized satellite arrays or file paths and formats them into 
        standardized PIL Images required by Hugging Face processors.

        Args:
            image_data: A Path to a generated tile on disk, or a 
                        normalized numpy array (3, H, W) directly from memory.

        Returns:
            A 3-channel RGB PIL Image ready for VLM prompt injection.
        """
        if isinstance(image_data, Path):
            if not image_data.exists():
                raise FileNotFoundError(f"Image tile not found at {image_data}")
            
            # PIL natively handles 8-bit GeoTIFFs exported by the chipper
            pil_image = Image.open(image_data).convert("RGB")
            
        elif isinstance(image_data, np.ndarray):
            # Normalizer outputs (Bands, Height, Width). PIL expects (Height, Width, Bands).
            if image_data.ndim == 3 and image_data.shape[0] == 3:
                transposed_array = np.transpose(image_data, (1, 2, 0))
            else:
                transposed_array = image_data

            # Enforce 8-bit unsigned integer type for PIL conversion
            if transposed_array.dtype != np.uint8:
                transposed_array = transposed_array.astype(np.uint8)

            pil_image = Image.fromarray(transposed_array, mode="RGB")
            
        else:
            raise TypeError(
                f"Expected pathlib.Path or numpy.ndarray, got {type(image_data)}"
            )

        return pil_image


    def format_prompt(self, text_query: str, task_prefix: str = "<CAPTION_TO_PHRASE>") -> str:
        """
        Step 3: Prompt Formatting
        Injects plain-language queries (e.g., 'damaged buildings', 'tents') into the 
        specific syntax expected by the VLM. Defaulting to Florence-2's phrase grounding.

        Args:
            text_query: The plain text description of the object to detect.
            task_prefix: The model-specific task token (e.g., <OD>, <CAPTION_TO_PHRASE>).

        Returns:
            Formatted string ready for the processor.
        """
        if not text_query:
            return task_prefix
        # Florence-2 expects the task token followed by the text query
        return f"{task_prefix} {text_query}"

    def predict(self, image: Image.Image, text_query: str) -> list[dict]:
        """
        Step 4: Offline Inference
        Executes the forward pass on the GPU in a strict no-grad context for memory efficiency.

        Args:
            image: Standardized PIL Image (from ingest_image).
            text_query: The plain text object to search for.

        Returns:
            List of standardized detection dictionaries.
        """
        prompt = self.format_prompt(text_query)
        
        # Process inputs and push to hardware specified in config
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
        
        # Cast pixel values to the memory-efficient precision
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(config.TORCH_DTYPE)

        # Execute forward pass without gradient tracking
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

        return self._standardize_output(generated_ids, image.size, task_prefix="<CAPTION_TO_PHRASE>")

    def _standardize_output(self, generated_ids: torch.Tensor, image_size: tuple, task_prefix: str) -> list[dict]:
        """
        Step 5: Output Standardization
        Parses raw model generation into a uniform data structure compatible with RasterRebuilder.

        Args:
            generated_ids: Raw output tensor from the model.
            image_size: Tuple of (width, height) used to scale bounding boxes back to pixel coordinates.
            task_prefix: The task token used, necessary for Hugging Face post-processing.

        Returns:
            List of dictionaries: [{"bbox": [xmin, ymin, xmax, ymax], "label": str, "confidence": float}]
        """
        # Decode token IDs back to text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Utilize processor's built-in parsing to extract coordinates based on original image size
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prefix, 
            image_size=image_size
        )
        
        standardized_results = []
        
        # Extract parsed results. Florence-2 returns a dict with task_prefix as the key
        detections = parsed_answer.get(task_prefix, {})
        bboxes = detections.get("bboxes", [])
        labels = detections.get("labels", [])
        
        for bbox, label in zip(bboxes, labels):
            standardized_results.append({
                "bbox": bbox,  # [xmin, ymin, xmax, ymax] in local tile pixel coordinates
                "label": label,
                "confidence": 1.0  # Florence-2 generation does not natively output confidence scores
            })
            
        return standardized_results