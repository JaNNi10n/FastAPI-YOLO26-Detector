"""Measure end-to-end latency of a running API: upload -> decode -> inference -> JSON.

Start the server first (one worker), then run e.g.:

    uvicorn app.main:app --port 8000
    python scripts/benchmark_api.py --label gpu --n 100

Client latency includes HTTP upload over localhost, so it is an upper bound on the
server-side cost. `inference_ms` is what the server reports for the model call alone.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 2)


def summary(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 2),
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "min": round(min(values), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--image", default=str(ROOT / "tests" / "assets" / "elephant_zebra.jpg"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--label", required=True, help="e.g. gpu or cpu; used in the output file name")
    args = p.parse_args()

    data = Path(args.image).read_bytes()
    files = {"file": (Path(args.image).name, data, "image/jpeg")}
    with httpx.Client(base_url=args.url, timeout=60) as client:
        health = client.get("/health").raise_for_status().json()
        for _ in range(args.warmup):
            client.post("/predict", files=files).raise_for_status()
        client_ms, server_ms = [], []
        for _ in range(args.n):
            t0 = time.perf_counter()
            r = client.post("/predict", files=files).raise_for_status()
            client_ms.append((time.perf_counter() - t0) * 1000)
            server_ms.append(r.json()["inference_ms"])

    report = {
        "label": args.label,
        "model": health["model"],
        "device": health["device"],
        "image": Path(args.image).name,
        "requests": args.n,
        "sequential_client_latency_ms": summary(client_ms),
        "server_inference_ms": summary(server_ms),
        "throughput_req_per_s": round(args.n / (sum(client_ms) / 1000), 1),
        "python": platform.python_version(),
        "os": platform.platform(),
    }
    out = ROOT / "results" / f"latency_{args.label}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
