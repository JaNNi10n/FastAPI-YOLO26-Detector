"""FastAPI service that serves a YOLO object detector over HTTP.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings
from app.model import Detector
from app.schemas import HealthResponse, ImageSize, PredictResponse

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs early (PIL raises above 2x this)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    detector = Detector(settings.model_path, device=settings.device, imgsz=settings.imgsz)
    app.state.settings = settings
    app.state.detector = detector
    app.state.device = detector.warmup()
    yield


app = FastAPI(
    title="YOLO Object Detection API",
    description="Upload an image, get back bounding boxes, class names and confidence scores.",
    version="1.0.0",
    lifespan=lifespan,
)


def load_image(file: UploadFile, max_bytes: int) -> Image.Image:
    """Validate an upload and decode it in memory (no temp files on disk)."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported content type {file.content_type!r}; send one of {sorted(ALLOWED_TYPES)}")
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File larger than {max_bytes // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)  # respect phone-camera rotation
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, f"Could not decode image: {exc}") from exc


# Query params shared by both predict endpoints.
Conf = Query(0.25, ge=0.0, le=1.0, description="Minimum confidence to keep a box")
Iou = Query(0.7, ge=0.0, le=1.0, description="IoU threshold for non-max suppression")
MaxDet = Query(100, ge=1, le=300, description="Maximum number of boxes returned")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    detector: Detector = request.app.state.detector
    return HealthResponse(
        status="ok", model=detector.name, device=request.app.state.device, classes=list(detector.names.values())
    )


# Plain `def` (not `async def`): inference is blocking, so FastAPI runs it in its thread pool
# instead of freezing the event loop for every other request.
@app.post("/predict", response_model=PredictResponse)
def predict(
    request: Request, file: UploadFile = File(...), conf: float = Conf, iou: float = Iou, max_det: int = MaxDet
) -> PredictResponse:
    image = load_image(file, request.app.state.settings.max_upload_bytes)
    detector: Detector = request.app.state.detector
    detections, ms = detector.predict(image, conf=conf, iou=iou, max_det=max_det)
    return PredictResponse(
        model=detector.name,
        image=ImageSize(width=image.width, height=image.height),
        detections=detections,
        inference_ms=ms,
    )


@app.post("/predict/annotated", response_class=Response, responses={200: {"content": {"image/jpeg": {}}}})
def predict_annotated(
    request: Request, file: UploadFile = File(...), conf: float = Conf, iou: float = Iou, max_det: int = MaxDet
) -> Response:
    """Same as /predict but returns the image with boxes drawn on it (JPEG)."""
    image = load_image(file, request.app.state.settings.max_upload_bytes)
    plotted = request.app.state.detector.annotate(image, conf=conf, iou=iou, max_det=max_det)
    ok, buf = cv2.imencode(".jpg", plotted)
    if not ok:
        raise HTTPException(500, "Could not encode annotated image")
    return Response(buf.tobytes(), media_type="image/jpeg")
