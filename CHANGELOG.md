# Changelog

## [1.0.0] - 2026-10-13

### Changed
- The default model is now **YOLO26n fine-tuned on the public African Wildlife dataset**
  (buffalo, elephant, rhino, zebra), trained with `scripts/train.py`. It is evaluated on the
  dataset's held-out test split (`results/metrics_test.json`).
- The original checkpoint (`best.pt`) is kept as `models/yolov8n-coco128-demo.pt`. Its embedded
  metadata shows YOLOv8n fine-tuned on coco128, not YOLO26 on a custom dataset as the first
  commit message said. Load it with `MODEL_PATH=models/yolov8n-coco128-demo.pt`.
- Code moved from a single `app.py` into an `app/` package (`config.py`, `model.py`, `schemas.py`,
  `main.py`). Start the server with `uvicorn app.main:app`.
- **Breaking:** `/predict` now returns
  `{model, image: {width, height}, detections: [{class_id, class_name, confidence, box}], inference_ms}`
  instead of `{boxes, classes}`. Confidence scores were missing before.
- `/health` returns the loaded model, device and class names.

### Fixed
- Concurrent requests overwrote each other's upload: every request wrote to the same `temp.jpg`.
  Uploads are now decoded in memory and nothing is written to disk.
- Non-image uploads crashed with HTTP 500 (`IndexError`). They now get 415 (wrong content type),
  400 (empty or undecodable file) or 413 (larger than `MAX_UPLOAD_MB`).
- `/predict` was `async def` but ran blocking inference, stalling the event loop. It is now a
  plain `def` (runs in FastAPI's thread pool), with a lock around the shared model.
- The weights path was relative to the working directory. It now defaults to a path relative to
  the package and can be overridden with `MODEL_PATH`.
- Phone photos with EXIF rotation are now rotated before inference.
- `/health` reported `cpu` while inference ran on the GPU; it now reports the predictor's real device.
- `scripts/evaluate.py`: Ultralytics' NMS time limit silently dropped whole batches on a busy GPU and
  deflated mAP (val mAP50 0.772 instead of 0.966). Evaluation now disables the limit; both metric files were re-run.
- Dockerfile: installs the system libraries OpenCV needs, uses CPU-only PyTorch, runs as a non-root
  user, adds a `HEALTHCHECK`, and has a `.dockerignore` so `.git` isn't copied into the image.

### Added
- `conf`, `iou` and `max_det` query parameters with validation (422 on out-of-range values).
- `POST /predict/annotated` returns the image with boxes drawn (JPEG).
- Pinned `requirements.txt` and `requirements-dev.txt`.
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/benchmark_api.py`, `scripts/inspect_checkpoint.py`.
- `results/` with training curves, held-out metrics and latency measurements.
- pytest suite (`tests/`) and a GitHub Actions workflow (ruff + pytest on CPU).
- README, MIT LICENSE, this changelog.

## [0.1.0] - 2026-09-22
- Initial version: single-file FastAPI app with `/health` and `/predict`, Dockerfile.
