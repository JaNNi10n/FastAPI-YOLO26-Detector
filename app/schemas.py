"""Pydantic response models (these also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Box(BaseModel):
    """Pixel coordinates in the original image: top-left (x1, y1), bottom-right (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    box: Box


class ImageSize(BaseModel):
    width: int
    height: int


class PredictResponse(BaseModel):
    model: str
    image: ImageSize
    detections: list[Detection]
    inference_ms: float = Field(description="Model time only (pre-process + forward + NMS), not network/upload")


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    classes: list[str]
