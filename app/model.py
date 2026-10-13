"""Thin wrapper around an Ultralytics YOLO model: load once, predict many times."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.schemas import Box, Detection


class Detector:
    def __init__(self, model_path: Path, device: str | None = None, imgsz: int = 640):
        from ultralytics import YOLO  # heavy import, kept local so app.schemas etc. stay light

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model weights not found: {model_path} (set MODEL_PATH)")
        self.model = YOLO(str(model_path))
        self.name = Path(model_path).name
        self.device = device
        self.imgsz = imgsz
        self.names: dict[int, str] = dict(self.model.names)
        # FastAPI runs sync endpoints in a thread pool, so two requests can hit this shared
        # model at once. YOLO.predict() writes the per-call options (conf, max_det, ...) onto
        # one shared predictor before running, so concurrent calls could use each other's
        # thresholds. One lock serialises inference; scale out with more uvicorn workers.
        self._lock = threading.Lock()

    def warmup(self) -> str:
        """Run one dummy inference so the first real request doesn't pay CUDA/cuDNN init cost."""
        self._run(Image.new("RGB", (self.imgsz, self.imgsz)), conf=0.25, iou=0.7, max_det=1)
        # The predictor holds the copy of the weights that actually runs; `self.model.device`
        # still reports the CPU copy loaded from disk, even when inference runs on CUDA.
        return str(self.model.predictor.device)

    def _run(self, image: Image.Image, conf: float, iou: float, max_det: int):
        """Run the model; returns (result, milliseconds spent in the model, excluding lock wait)."""
        with self._lock:
            start = time.perf_counter()
            result = self.model.predict(
                image, conf=conf, iou=iou, max_det=max_det, imgsz=self.imgsz, device=self.device, verbose=False
            )[0]
            return result, (time.perf_counter() - start) * 1000

    def predict(
        self, image: Image.Image, conf: float = 0.25, iou: float = 0.7, max_det: int = 100
    ) -> tuple[list[Detection], float]:
        result, elapsed_ms = self._run(image, conf, iou, max_det)

        boxes = result.boxes
        xyxy = boxes.xyxy.tolist()
        detections = []
        for (x1, y1, x2, y2), cls, conf_ in zip(xyxy, boxes.cls.tolist(), boxes.conf.tolist(), strict=True):
            detections.append(
                Detection(
                    class_id=int(cls),
                    class_name=self.names[int(cls)],
                    confidence=round(conf_, 4),
                    box=Box(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1)),
                )
            )
        return detections, round(elapsed_ms, 2)

    def annotate(self, image: Image.Image, conf: float = 0.25, iou: float = 0.7, max_det: int = 100) -> np.ndarray:
        """Return a BGR numpy image with boxes and labels drawn on it."""
        return self._run(image, conf, iou, max_det)[0].plot()
