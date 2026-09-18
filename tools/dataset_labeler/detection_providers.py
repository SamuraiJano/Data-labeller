from __future__ import annotations

import inspect
import importlib.util
import os
import threading
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_MODEL_URL = f"https://huggingface.co/{GROUNDING_DINO_MODEL_ID}"
GROUNDING_DINO_SOURCE = "https://github.com/IDEA-Research/GroundingDINO"


def _iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(
    proposals: list[dict[str, Any]], threshold: float = 0.55, maximum: int = 100
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=lambda item: float(item["score"]), reverse=True):
        if all(_iou(proposal["bbox"], existing["bbox"]) < threshold for existing in kept):
            kept.append(proposal)
        if len(kept) >= maximum:
            break
    return kept


class MockDetectionProvider:
    name = "mock"

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "state": "ready",
            "error": None,
            "model_id": "mock-citrus-boxes",
            "model_url": None,
            "official_repository": None,
            "license": "test-only",
            "device": "cpu",
            "capabilities": {"text_prompts": True, "multiple_boxes": True},
        }

    def load(self) -> dict[str, Any]:
        return self.status()

    def predict(self, image_path: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
        with Image.open(image_path) as opened:
            width, height = ImageOps.exif_transpose(opened).size
        label = str((settings.get("labels") or ["orange"])[0])
        return [
            {
                "id": "mock-box-1",
                "label": label,
                "score": 0.91,
                "bbox": [width * 0.18, height * 0.16, width * 0.42, height * 0.58],
                "source": "mock",
                "model_id": "mock-citrus-boxes",
            },
            {
                "id": "mock-box-2",
                "label": label,
                "score": 0.78,
                "bbox": [width * 0.58, height * 0.28, width * 0.28, height * 0.45],
                "source": "mock",
                "model_id": "mock-citrus-boxes",
            },
        ][: int(settings.get("max_detections", 25))]


class GroundingDinoDetectionProvider:
    name = "grounding_dino"

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.state = "unavailable"
        self.error: str | None = None
        self.device = os.getenv("GROUNDING_DINO_DEVICE", "auto").strip().lower()
        self.model_id = os.getenv("GROUNDING_DINO_MODEL_ID", GROUNDING_DINO_MODEL_ID).strip()
        missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
        if not missing:
            self.state = "idle"
            self.error = "Model support is installed but the detector has not been loaded yet."
        else:
            self.error = (
                "Grounding DINO support is not installed. Restart with start.cmd to install it, "
                f"or run python -m pip install -e \".[detection]\". Missing: {', '.join(missing)}."
            )

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "state": self.state,
            "error": self.error,
            "model_id": self.model_id,
            "model_url": f"https://huggingface.co/{self.model_id}",
            "official_repository": GROUNDING_DINO_SOURCE,
            "license": "Apache-2.0",
            "device": self.device,
            "capabilities": {"text_prompts": True, "multiple_boxes": True},
        }

    def _resolved_device(self, torch: Any) -> str:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise RuntimeError("GROUNDING_DINO_DEVICE must be 'auto', 'cpu', or 'cuda'.")
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Set GROUNDING_DINO_DEVICE=cpu or use auto selection.")
        return self.device

    def load(self) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            self.state = "ready"
            self.error = None
            return self.status()
        self.state = "loading"
        self.error = None
        try:
            if os.name == "nt":
                import truststore

                # Hugging Face downloads still verify TLS, but use the same
                # Windows trust roots as the browser and modern pip.
                truststore.inject_into_ssl()
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.device = self._resolved_device(torch)
            self.torch = torch
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
            self.state = "ready"
        except Exception as exc:
            self.model = None
            self.processor = None
            self.state = "failed"
            self.error = f"Grounding DINO could not be loaded: {exc}"
        return self.status()

    def predict(self, image_path: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
        if self.model is None or self.processor is None:
            status = self.load()
            if status["state"] != "ready":
                raise RuntimeError(status.get("error") or "Grounding DINO could not be loaded.")
        labels = [str(value).strip().lower() for value in settings.get("labels", []) if str(value).strip()]
        if not labels:
            raise ValueError("Enter at least one object description, such as orange or naartjie.")
        confidence = float(settings.get("confidence", 0.30))
        text_threshold = float(settings.get("text_threshold", 0.25))
        maximum = int(settings.get("max_detections", 25))
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        inputs = self.processor(images=image, text=[labels], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        post_process = self.processor.post_process_grounded_object_detection
        parameters = inspect.signature(post_process).parameters
        threshold_arguments = (
            {"threshold": confidence, "text_threshold": text_threshold}
            if "threshold" in parameters
            else {"box_threshold": confidence, "text_threshold": text_threshold}
        )
        result = post_process(
            outputs,
            inputs["input_ids"],
            target_sizes=[image.size[::-1]],
            **threshold_arguments,
        )[0]
        result_labels = result.get("text_labels")
        if result_labels is None:
            result_labels = result.get("labels", [])
        proposals: list[dict[str, Any]] = []
        for index, (box_value, score_value, label_value) in enumerate(
            zip(result.get("boxes", []), result.get("scores", []), result_labels, strict=False)
        ):
            x1, y1, x2, y2 = [float(value) for value in box_value.detach().cpu().tolist()]
            x1 = max(0.0, min(float(image.width), x1))
            y1 = max(0.0, min(float(image.height), y1))
            x2 = max(x1, min(float(image.width), x2))
            y2 = max(y1, min(float(image.height), y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            raw_label = label_value.item() if hasattr(label_value, "item") else label_value
            label = labels[int(raw_label)] if isinstance(raw_label, int) and 0 <= raw_label < len(labels) else str(raw_label)
            score = float(score_value.detach().cpu().item() if hasattr(score_value, "detach") else score_value)
            proposals.append(
                {
                    "id": f"grounding-dino-{index + 1}",
                    "label": label,
                    "score": score,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "source": "grounding_dino",
                    "model_id": self.model_id,
                }
            )
        self.state = "ready"
        self.error = None
        return non_maximum_suppression(proposals, maximum=maximum)


class DetectionProviderManager:
    def __init__(self, provider_name: str | None = None) -> None:
        name = (provider_name or os.getenv("JANO_DETECTION_PROVIDER") or "grounding_dino").strip().lower()
        if name == "mock":
            self.provider: Any = MockDetectionProvider()
        elif name in {"grounding_dino", "grounding-dino"}:
            self.provider = GroundingDinoDetectionProvider()
        else:
            raise ValueError(f"Unknown detection provider '{name}'. Expected 'mock' or 'grounding_dino'.")
        self.lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return self.provider.status()

    def load(self) -> dict[str, Any]:
        with self.lock:
            return self.provider.load()

    def predict(self, image_path: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
        with self.lock:
            return self.provider.predict(image_path, settings)
