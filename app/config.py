"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "yolo26n-african-wildlife.pt"


@dataclass(frozen=True)
class Settings:
    model_path: Path
    device: str | None  # None = let Ultralytics pick (GPU if available)
    max_upload_bytes: int
    imgsz: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", DEFAULT_MODEL)),
            device=os.getenv("DEVICE") or None,
            max_upload_bytes=int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024),
            imgsz=int(os.getenv("IMGSZ", "640")),
        )
