from __future__ import annotations

import base64
import hashlib
import io
import os
import platform
import subprocess
import threading
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

SAM2_OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/sam2"
SAM2_PINNED_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-small"
SAM2_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "sam2.1_hiera_small.pt"
SAM2_DEFAULT_CHECKPOINT_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
SAM3_OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/sam3"
SAM3_PINNED_REVISION = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"
SAM3_DEFAULT_MODEL_ID = "facebook/sam3"


def mask_data_url(mask: np.ndarray) -> str:
    binary = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    output = io.BytesIO()
    binary.save(output, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SegmentationProvider(ABC):
    name = "base"

    def __init__(self) -> None:
        self.state = "unavailable"
        self.error: str | None = None
        self.peak_vram_mb: float | None = None

    def status(self) -> dict[str, Any]:
        is_sam2 = self.name == "sam2"
        is_sam3 = self.name == "sam3"
        configured_checkpoint = os.getenv("SAM2_CHECKPOINT_PATH", "").strip()
        checkpoint_path = configured_checkpoint or (str(SAM2_DEFAULT_CHECKPOINT) if SAM2_DEFAULT_CHECKPOINT.is_file() else None)
        return {
            "provider": self.name,
            "state": self.state,
            "error": self.error,
            "peak_vram_mb": self.peak_vram_mb,
            "device": getattr(self, "device", None),
            "capabilities": self.capabilities(),
            "official_repository": SAM2_OFFICIAL_REPOSITORY if is_sam2 else SAM3_OFFICIAL_REPOSITORY if is_sam3 else None,
            "pinned_revision": SAM2_PINNED_REVISION if is_sam2 else SAM3_PINNED_REVISION if is_sam3 else None,
            "model_id": (
                os.getenv("SAM2_MODEL_ID", SAM2_DEFAULT_MODEL_ID)
                if is_sam2
                else os.getenv("SAM3_MODEL_ID", SAM3_DEFAULT_MODEL_ID) if is_sam3 else None
            ),
            "checkpoint_path": checkpoint_path if is_sam2 else (os.getenv("SAM3_CHECKPOINT_PATH", "").strip() or None) if is_sam3 else None,
        }

    @abstractmethod
    def capabilities(self) -> dict[str, bool]:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_path: Path, prompts: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def clear_image(self) -> None:
        return None


class MockSegmentationProvider(SegmentationProvider):
    name = "mock"

    def __init__(self) -> None:
        super().__init__()
        self.state = "ready"

    def capabilities(self) -> dict[str, bool]:
        return {"positive_points": True, "negative_points": True, "box": True, "text": True, "multiple_proposals": True}

    def load(self) -> dict[str, Any]:
        self.state = "ready"
        return self.status()

    def predict(self, image_path: Path, prompts: dict[str, Any]) -> list[dict[str, Any]]:
        with Image.open(image_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        points = prompts.get("points") or []
        box = prompts.get("box")
        variants: list[tuple[np.ndarray, float]] = []
        for scale, score in ((1.0, 0.92), (0.88, 0.84), (1.12, 0.76)):
            mask = np.zeros((height, width), dtype=np.uint8)
            if box:
                x1, y1, x2, y2 = [float(value) for value in box]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                half_width, half_height = abs(x2 - x1) * scale / 2, abs(y2 - y1) * scale / 2
                cv2.ellipse(mask, (round(cx), round(cy)), (max(1, round(half_width)), max(1, round(half_height))), 0, 0, 360, 255, -1)
            else:
                positives = [point for point in points if int(point.get("label", 1)) == 1]
                center = positives[0] if positives else {"x": width / 2, "y": height / 2}
                radius = max(8, round(min(width, height) * 0.18 * scale))
                cv2.circle(mask, (round(float(center["x"])), round(float(center["y"]))), radius, 255, -1)
            for point in points:
                radius = max(5, round(min(width, height) * 0.06))
                color = 255 if int(point.get("label", 1)) == 1 else 0
                cv2.circle(mask, (round(float(point["x"])), round(float(point["y"]))), radius, color, -1)
            variants.append((mask > 0, score))
        return [
            {"id": f"mock-{index + 1}", "score": score, "mask_data": mask_data_url(mask)}
            for index, (mask, score) in enumerate(variants)
        ]


class Sam2SegmentationProvider(SegmentationProvider):
    """Lazy adapter around Meta's official SAM 2.1 image predictor."""

    name = "sam2"

    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.predictor: Any = None
        self.current_image_key: str | None = None
        self.torch: Any = None
        self.device = os.getenv("SAM2_DEVICE", "auto").strip().lower()
        try:
            import sam2  # noqa: F401
            import torch  # noqa: F401

            self.state = "idle"
            self.error = "Model is installed but not loaded."
        except Exception as exc:
            self.error = f"SAM 2 is not installed in this Python environment: {exc}"

    def capabilities(self) -> dict[str, bool]:
        return {"positive_points": True, "negative_points": True, "box": True, "text": False, "multiple_proposals": True}

    def _installed_revision(self) -> str | None:
        repository = os.getenv("SAM2_REPO_PATH")
        if not repository:
            return None
        try:
            return subprocess.check_output(
                ["git", "-C", str(Path(repository).expanduser()), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except Exception:
            return None

    def _resolved_device(self, torch: Any) -> str:
        if self.device not in {"cuda", "cpu", "auto"}:
            raise RuntimeError("SAM2_DEVICE must be 'cuda', 'cpu', or 'auto'.")
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Set SAM2_DEVICE=cpu for slow CPU inference, or use the manual/mock provider."
            )
        return device

    def load(self) -> dict[str, Any]:
        if self.predictor is not None:
            self.state = "ready"
            self.error = None
            return self.status()
        self.state = "loading"
        self.error = None
        try:
            import torch
            from sam2.build_sam import build_sam2, build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = self._resolved_device(torch)
            self.torch = torch
            self.device = device
            installed_revision = self._installed_revision()
            requested_revision = os.getenv("SAM2_REVISION", SAM2_PINNED_REVISION)
            if installed_revision and installed_revision != requested_revision:
                raise RuntimeError(
                    f"SAM2_REPO_PATH is at {installed_revision}, but this app is pinned to {requested_revision}. "
                    "Checkout the pinned revision and reinstall the package."
                )

            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            checkpoint_value = os.getenv("SAM2_CHECKPOINT_PATH", "").strip()
            checkpoint = Path(checkpoint_value).expanduser().resolve() if checkpoint_value else SAM2_DEFAULT_CHECKPOINT
            if checkpoint_value and not checkpoint.is_file():
                raise RuntimeError(f"SAM2_CHECKPOINT_PATH does not exist: {checkpoint}")
            if checkpoint.is_file():
                expected_hash = os.getenv("SAM2_CHECKPOINT_SHA256", "").strip().lower()
                if not expected_hash and checkpoint.resolve() == SAM2_DEFAULT_CHECKPOINT.resolve():
                    expected_hash = SAM2_DEFAULT_CHECKPOINT_SHA256
                if expected_hash:
                    actual_hash = file_sha256(checkpoint)
                    if actual_hash != expected_hash:
                        raise RuntimeError(
                            f"SAM 2 checkpoint checksum mismatch for {checkpoint}. "
                            f"Expected {expected_hash}, received {actual_hash}."
                        )
                model_config = os.getenv("SAM2_MODEL_CONFIG", SAM2_DEFAULT_CONFIG).strip()
                self.model = build_sam2(model_config, str(checkpoint), device=device)
            else:
                model_id = os.getenv("SAM2_MODEL_ID", SAM2_DEFAULT_MODEL_ID).strip()
                self.model = build_sam2_hf(model_id, device=device)

            self.predictor = SAM2ImagePredictor(self.model)
            self.state = "ready"
            self.error = None
            self.peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if device == "cuda" else None
        except Exception as exc:
            self.model = None
            self.predictor = None
            self.state = "failed"
            if self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
                self.torch.cuda.empty_cache()
                self.error = "CUDA ran out of memory while loading SAM 2.1. Use the small/tiny checkpoint or close other GPU applications."
            else:
                self.error = str(exc)
        return self.status()

    def _set_image(self, image_path: Path) -> None:
        key = f"{image_path}:{image_path.stat().st_mtime_ns}"
        if key == self.current_image_key:
            return
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            self.predictor.set_image(image)
        self.current_image_key = key

    def predict(self, image_path: Path, prompts: dict[str, Any]) -> list[dict[str, Any]]:
        text = str(prompts.get("text") or "").strip()
        if text:
            raise RuntimeError("SAM 2 does not support text prompts. Use positive/negative points or a box.")
        points = prompts.get("points") or []
        box = prompts.get("box")
        if not points and not box:
            raise RuntimeError("Add at least one positive/negative point or a box before requesting a SAM 2 proposal.")
        if self.predictor is None:
            status = self.load()
            if status["state"] != "ready":
                raise RuntimeError(status.get("error") or "SAM 2 could not be loaded.")

        self.state = "busy"
        try:
            self._set_image(image_path)
            point_coords = (
                np.asarray([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float32)
                if points
                else None
            )
            point_labels = (
                np.asarray([int(point.get("label", 1)) for point in points], dtype=np.int32)
                if points
                else None
            )
            input_box = np.asarray(box, dtype=np.float32) if box else None
            autocast = (
                self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                if self.device == "cuda"
                else nullcontext()
            )
            with self.torch.inference_mode(), autocast:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=input_box,
                    multimask_output=True,
                )

            flat_scores = np.asarray(scores).reshape(-1)
            masks_array = np.asarray(masks)
            order = np.argsort(flat_scores)[::-1]
            proposals = [
                {
                    "id": f"sam2-{index + 1}",
                    "score": float(flat_scores[position]),
                    "mask_data": mask_data_url(masks_array[position] > 0),
                }
                for index, position in enumerate(order[:3])
            ]
            if self.device == "cuda":
                self.peak_vram_mb = float(self.torch.cuda.max_memory_allocated() / (1024**2))
            self.state = "ready"
            self.error = None
            return proposals
        except Exception as exc:
            if self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
                self.clear_image()
                self.error = (
                    "CUDA ran out of memory. Close other GPU applications or select a smaller SAM 2.1 checkpoint, "
                    "then retry. Manual and mock annotation remain available."
                )
            else:
                self.error = str(exc)
            self.state = "failed"
            raise RuntimeError(self.error) from exc

    def clear_image(self) -> None:
        self.current_image_key = None
        if self.predictor is not None:
            self.predictor.reset_predictor()
        if self.torch is not None and self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class Sam3SegmentationProvider(SegmentationProvider):
    """Lazy adapter around Meta's official facebookresearch/sam3 package."""

    name = "sam3"

    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.image_state: dict[str, Any] | None = None
        self.current_image_key: str | None = None
        self.device = os.getenv("SAM3_DEVICE", "auto").strip().lower()
        try:
            import sam3  # noqa: F401
            import torch  # noqa: F401

            self.state = "idle"
            self.error = "Official SAM 3 package is installed but the model is not loaded."
        except Exception as exc:
            self.error = (
                "Official Meta SAM 3 is not installed in this Python environment. "
                f"Install facebookresearch/sam3 in editable mode; manual annotation remains available. Details: {exc}"
            )

    def capabilities(self) -> dict[str, bool]:
        return {"positive_points": False, "negative_points": False, "box": True, "text": True, "multiple_proposals": True}

    def _installed_revision(self) -> str | None:
        repository = os.getenv("SAM3_REPO_PATH", "").strip()
        if not repository:
            return None
        try:
            return subprocess.check_output(
                ["git", "-C", str(Path(repository).expanduser()), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except Exception:
            return None

    def _resolved_device(self, torch: Any) -> str:
        if platform.machine().lower() in {"aarch64", "arm64", "armv7l", "armv8l"}:
            raise RuntimeError("SAM 3 is a workstation annotation aid and is deliberately disabled on Raspberry Pi/ARM hosts.")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise RuntimeError("SAM3_DEVICE must be 'cuda', 'cpu', or 'auto'.")
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Set SAM3_DEVICE=cpu for a slow workstation smoke test or continue manual labelling.")
        return device

    def load(self) -> dict[str, Any]:
        if self.model is not None and self.processor is not None:
            self.state = "ready"
            self.error = None
            return self.status()
        self.state = "loading"
        self.error = None
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
            if version < (2, 7):
                raise RuntimeError(
                    f"Official SAM 3 requires PyTorch >=2.7; found {torch.__version__}. "
                    "Use a separate supported environment or continue with manual annotation."
                )
            device = self._resolved_device(torch)
            self.torch = torch
            self.device = device
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            installed_revision = self._installed_revision()
            requested_revision = os.getenv("SAM3_REVISION", SAM3_PINNED_REVISION).strip()
            if installed_revision and requested_revision and installed_revision != requested_revision:
                raise RuntimeError(
                    f"SAM3_REPO_PATH is at {installed_revision}, expected {requested_revision}. "
                    "Checkout the configured revision and reinstall the official package."
                )
            checkpoint_value = os.getenv("SAM3_CHECKPOINT_PATH", "").strip()
            checkpoint = Path(checkpoint_value).expanduser().resolve() if checkpoint_value else None
            if checkpoint is not None and not checkpoint.is_file():
                raise RuntimeError(f"SAM3_CHECKPOINT_PATH does not exist: {checkpoint}")
            load_from_hf = os.getenv("SAM3_LOAD_FROM_HF", "true").strip().lower() in {"1", "true", "yes"}
            if checkpoint is None and not load_from_hf:
                raise RuntimeError("Configure SAM3_CHECKPOINT_PATH or set SAM3_LOAD_FROM_HF=true.")
            self.model = build_sam3_image_model(
                device=device,
                checkpoint_path=str(checkpoint) if checkpoint else None,
                load_from_HF=load_from_hf,
                eval_mode=True,
            )
            threshold = float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.35"))
            self.processor = Sam3Processor(self.model, device=device, confidence_threshold=threshold)
            self.state = "ready"
            self.error = None
            self.peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if device == "cuda" else None
        except Exception as exc:
            self.model = None
            self.processor = None
            self.state = "failed"
            if self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
                self.torch.cuda.empty_cache()
                self.error = "CUDA ran out of memory while loading SAM 3. Close other GPU applications and retry; manual labelling remains available."
            else:
                self.error = f"{exc} Manual annotation remains available."
        return self.status()

    def predict(self, image_path: Path, prompts: dict[str, Any]) -> list[dict[str, Any]]:
        text = str(prompts.get("text") or "").strip()
        points = prompts.get("points") or []
        box = prompts.get("box")
        if points:
            raise RuntimeError("Meta's official SAM3 image processor supports text and box prompts here; use a box or text, or continue manually.")
        if not text and not box:
            raise RuntimeError("Add a text concept or box before requesting a SAM 3 proposal.")
        if self.model is None or self.processor is None:
            status = self.load()
            if status["state"] != "ready":
                raise RuntimeError(status.get("error") or "SAM 3 could not be loaded.")

        self.state = "busy"
        try:
            key = f"{image_path}:{image_path.stat().st_mtime_ns}"
            if key != self.current_image_key or self.image_state is None:
                with Image.open(image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    self.image_state = self.processor.set_image(image)
                self.current_image_key = key
            else:
                image = None
            state = self.image_state
            self.processor.reset_all_prompts(state)
            if text:
                state = self.processor.set_text_prompt(text, state)
            if box:
                width = float(state["original_width"])
                height = float(state["original_height"])
                x1, y1, x2, y2 = [float(value) for value in box]
                normalized_box = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), abs(x2 - x1) / width, abs(y2 - y1) / height]
                state = self.processor.add_geometric_prompt(normalized_box, True, state)
            masks_tensor = state.get("masks")
            scores_tensor = state.get("scores")
            if masks_tensor is None or not int(masks_tensor.shape[0]):
                raise RuntimeError("SAM 3 returned no masks for this prompt. Refine the prompt or continue manually.")
            masks_array = masks_tensor.detach().cpu().numpy().astype(bool).reshape((-1, int(state["original_height"]), int(state["original_width"])))
            scores = scores_tensor.detach().float().cpu().numpy().reshape(-1) if scores_tensor is not None else np.zeros(len(masks_array))
            order = np.argsort(scores)[::-1] if scores.size else np.arange(masks_array.shape[0])
            proposals = [
                {
                    "id": f"sam3-{index + 1}",
                    "score": float(scores[position]) if position < scores.size else 0.0,
                    "mask_data": mask_data_url(masks_array[position]),
                }
                for index, position in enumerate(order[:3])
            ]
            if self.device == "cuda":
                self.peak_vram_mb = float(self.torch.cuda.max_memory_allocated() / (1024**2))
            self.state = "ready"
            self.error = None
            return proposals
        except Exception as exc:
            if self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
                self.torch.cuda.empty_cache()
                self.error = "CUDA ran out of memory during SAM 3 inference. Close other GPU applications; manual labelling remains available."
            else:
                self.error = str(exc)
            self.state = "failed"
            raise RuntimeError(self.error) from exc

    def clear_image(self) -> None:
        self.image_state = None
        self.current_image_key = None
        if self.torch is not None and self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class ProviderManager:
    def __init__(self, provider_name: str | None = None) -> None:
        # Never present the synthetic mock proposer as the production default.
        # SAM 2.1 uses the bundled, checksum-verified small checkpoint and can be
        # overridden explicitly for SAM 3 or UI development.
        name = (provider_name or os.getenv("JANO_SEGMENTATION_PROVIDER") or "sam2").strip().lower()
        if name == "mock":
            self.provider: SegmentationProvider = MockSegmentationProvider()
        elif name == "sam2":
            self.provider = Sam2SegmentationProvider()
        elif name == "sam3":
            self.provider = Sam3SegmentationProvider()
        else:
            raise ValueError(f"Unknown segmentation provider '{name}'. Expected 'mock', 'sam2', or 'sam3'.")
        self.lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return self.provider.status()

    def load(self) -> dict[str, Any]:
        with self.lock:
            return self.provider.load()

    def predict(self, image_path: Path, prompts: dict[str, Any]) -> list[dict[str, Any]]:
        with self.lock:
            return self.provider.predict(image_path, prompts)

    def clear_image(self) -> None:
        with self.lock:
            self.provider.clear_image()
