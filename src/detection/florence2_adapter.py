"""
src/detection/florence2_adapter.py

Florence-2 Vision-Language Model Inference Adapter.

Provides offline, air-gapped integration for Microsoft Florence-2 models. Loads local 
weights from config.MODELS_DIR / "florence2" and executes multi-task prompt queries 
(Open-Vocabulary Detection, Phrase Grounding) over tiled image inputs.
"""

from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class Florence2Adapter:
    """
    Adapter for running local, air-gapped Florence-2 Vision-Language Model tasks.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize and load Florence-2 model and processor from local configuration paths.

        Args:
            model_dir (Optional[Path]): Directory containing local Florence-2 model weights.
        """
        self.model_dir = model_dir or (config.MODELS_DIR / "florence2")
        self.device = config.DEVICE
        self.torch_dtype = config.TORCH_DTYPE

        # Load offline model weights and processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, 
            local_files_only=True,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

    def predict_tile(
        self,
        image_input: Union[str, Path, Image.Image],
        task_prompt: str = "<OPEN_VOCABULARY_DETECTION>",
        text_input: str = "military compound . guard tower . trench line",
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> pd.DataFrame:
        """
        Runs vision-language inference on a single tile or image path.

        Args:
            image_input (Union[str, Path, Image.Image]): PIL Image or image file path.
            task_prompt (str): Task identifier prompt (e.g., '<OPEN_VOCABULARY_DETECTION>').
            text_input (str): Target object queries separated by periods or commas.
            x_offset (int): X-coordinate offset for tile mosaic reconstruction.
            y_offset (int): Y-coordinate offset for tile mosaic reconstruction.

        Returns:
            pd.DataFrame: Detection records with pixel coordinates and labels.
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        prompt = task_prompt + text_input if text_input else task_prompt

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            device=self.device, dtype=self.torch_dtype
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_text = self.processor.batch_decode(
            generated_ids.sequences, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return self._format_parsed_output(parsed_answer, task_prompt, x_offset, y_offset)

    def _format_parsed_output(
        self,
        parsed_data: Dict[str, Any],
        task_prompt: str,
        x_offset: int,
        y_offset: int
    ) -> pd.DataFrame:
        """
        Formats raw model output dictionaries into standard Pandas DataFrames.
        """
        records = []

        if task_prompt in parsed_data:
            data = parsed_data[task_prompt]
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [])

            for bbox, label in zip(bboxes, labels):
                # Florence-2 box format: [xmin, ymin, xmax, ymax]
                xmin = float(bbox[0]) + x_offset
                ymin = float(bbox[1]) + y_offset
                xmax = float(bbox[2]) + x_offset
                ymax = float(bbox[3]) + y_offset

                records.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": str(label).strip(),
                    "confidence": 0.85,  # Baseline score placeholder for generation outputs
                })

        # Feature 4: Clear VRAM cache after execution
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pd.DataFrame(records)

    def predict_batch(
        self,
        tile_manifest: List[Dict[str, Any]],
        task_prompt: str = "<OPEN_VOCABULARY_DETECTION>",
        text_input: str = "military compound . guard tower"
    ) -> pd.DataFrame:
        """
        Feature 2: Processes a batch list of tile paths and offsets.

        Args:
            tile_manifest (List[Dict[str, Any]]): List of dicts containing 'path', 'x_offset', 'y_offset'.
            task_prompt (str): Task identifier prompt.
            text_input (str): Target detection query string.

        Returns:
            pd.DataFrame: Consolidated batch detections.
        """
        batch_results = []
        for item in tile_manifest:
            df = self.predict_tile(
                image_input=item["path"],
                task_prompt=task_prompt,
                text_input=text_input,
                x_offset=item.get("x_offset", 0),
                y_offset=item.get("y_offset", 0)
            )
            if not df.empty:
                batch_results.append(df)

        if not batch_results:
            return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "label", "confidence"])

        return pd.concat(batch_results, ignore_index=True)