"""Print (and optionally save) the training metadata stored inside an Ultralytics checkpoint.

    python scripts/inspect_checkpoint.py models/yolov8n-coco128-demo.pt --out results/legacy_checkpoint.json

Ultralytics stores the train args and the last validation metrics in every .pt
file, so anyone can check what a checkpoint really is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ("model", "data", "epochs", "imgsz", "batch", "name", "seed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", help="optional JSON output path")
    args = p.parse_args()

    import torch

    # weights_only=False: Ultralytics checkpoints pickle the model object. Only load files you trust.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args", {})
    info = {
        "checkpoint": Path(args.checkpoint).name,
        "saved_at": ckpt.get("date"),
        "ultralytics_version": ckpt.get("version"),
        "license": ckpt.get("license"),
        "train_args": {k: train_args.get(k) for k in KEYS},
        "num_classes": len(ckpt["model"].names),
        "class_names": list(ckpt["model"].names.values()),
        "last_val_metrics": ckpt.get("train_metrics"),
    }
    text = json.dumps(info, indent=2, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
