import dataclasses
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from PIL import Image

from app.model import Detector


def post(client, data: bytes, content_type="image/jpeg", path="/predict", **params):
    return client.post(path, files={"file": ("img", data, content_type)}, params=params)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["classes"] == ["buffalo", "elephant", "rhino", "zebra"]
    # Regression: /health used to report "cpu" even while inference ran on the GPU.
    on_gpu = torch.cuda.is_available() and os.getenv("DEVICE", "") != "cpu"
    assert body["device"].startswith("cuda" if on_gpu else "cpu")


def test_predict_finds_elephant_and_zebra(client, sample_bytes):
    r = post(client, sample_bytes)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] == {"width": 620, "height": 387}
    names = {d["class_name"] for d in body["detections"]}
    assert {"elephant", "zebra"} <= names
    for d in body["detections"]:
        b = d["box"]
        assert 0 <= b["x1"] < b["x2"] <= 620
        assert 0 <= b["y1"] < b["y2"] <= 387
        assert 0.25 <= d["confidence"] <= 1


def test_predict_finds_buffalo(client):
    r = post(client, (Path(__file__).parent / "assets" / "buffalo.jpg").read_bytes())
    assert r.status_code == 200, r.text
    assert "buffalo" in {d["class_name"] for d in r.json()["detections"]}


def test_conf_threshold_filters_boxes(client, sample_bytes):
    low = post(client, sample_bytes, conf=0.1).json()["detections"]
    high = post(client, sample_bytes, conf=0.8).json()["detections"]
    assert len(high) <= len(low)
    assert all(d["confidence"] >= 0.8 for d in high)


def test_png_with_alpha_is_accepted(client, sample_bytes):
    buf = io.BytesIO()
    Image.open(io.BytesIO(sample_bytes)).convert("RGBA").save(buf, format="PNG")
    r = post(client, buf.getvalue(), content_type="image/png")
    assert r.status_code == 200
    assert r.json()["detections"]


def test_annotated_returns_jpeg_of_same_size(client, sample_bytes):
    r = post(client, sample_bytes, path="/predict/annotated")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == (620, 387)


def test_rejects_wrong_content_type(client):
    r = post(client, b"hello", content_type="text/plain")
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = post(client, b"definitely not a jpeg")
    assert r.status_code == 400
    assert "decode" in r.json()["detail"]


def test_rejects_empty_file(client):
    assert post(client, b"").status_code == 400


def test_rejects_oversized_file(client, sample_bytes):
    original = client.app.state.settings
    client.app.state.settings = dataclasses.replace(original, max_upload_bytes=1000)
    try:
        assert post(client, sample_bytes).status_code == 413
    finally:
        client.app.state.settings = original


@pytest.mark.parametrize("params", [{"conf": 1.5}, {"iou": -0.1}, {"max_det": 0}])
def test_rejects_bad_query_params(client, sample_bytes, params):
    assert post(client, sample_bytes, **params).status_code == 422


def test_missing_file_field(client):
    assert client.post("/predict").status_code == 422


def test_concurrent_requests_get_consistent_results(client, sample_bytes):
    # The original app wrote every upload to one shared temp.jpg, so parallel requests
    # could read each other's image. Uploads are now decoded in memory per request.
    blank = io.BytesIO()
    Image.new("RGB", (320, 240), "white").save(blank, format="JPEG")

    def call(i):
        data = sample_bytes if i % 2 == 0 else blank.getvalue()
        return i, post(client, data).json()

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(call, range(12)))
    for i, body in results:
        expected = (620, 387) if i % 2 == 0 else (320, 240)
        assert (body["image"]["width"], body["image"]["height"]) == expected
        if i % 2:
            assert body["detections"] == []


def test_missing_weights_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="MODEL_PATH"):
        Detector(tmp_path / "nope.pt")
