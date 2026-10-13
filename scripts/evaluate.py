"""Evaluate a checkpoint on a dataset split and save the metrics as JSON.

    python scripts/evaluate.py --split test   # held-out split, never used for training or model selection

Writes results/metrics_<split>.json with overall and per-class precision, recall,
mAP50 and mAP50-95, plus the hardware and library versions used.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(ROOT / "models" / "yolo26n-african-wildlife.pt"))
    p.add_argument("--data", default="african-wildlife.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="dataloader worker processes; 0 = load in the main process (each worker re-imports torch, "
        "which ran out of page-file memory on the 16 GB dev machine)",
    )
    p.add_argument("--out", default=None, help="default: results/metrics_<split>.json")
    args = p.parse_args()

    import functools

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import nms

    # Ultralytics' NMS gives up after (2 s + 0.05 s x batch) and silently DROPS the predictions of
    # the remaining images in that batch ("WARNING NMS time limit ... exceeded"). On a busy or
    # shared GPU that happens often and deflates mAP badly (it did on the first run of this
    # script). Evaluation must score every image, so the limit is disabled here.
    nms.non_max_suppression = functools.partial(nms.non_max_suppression, max_time_img=1e6)

    model = YOLO(args.weights)
    m = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs"),
        name=f"eval-{args.split}",
        exist_ok=True,
        plots=True,
    )

    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p_, r_, ap50, ap = m.box.class_result(i)
        per_class[model.names[int(c)]] = {
            "precision": round(float(p_), 4),
            "recall": round(float(r_), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50-95": round(float(ap), 4),
            "instances": int(m.nt_per_class[int(c)]),
        }
    on_gpu = torch.cuda.is_available() and args.device != "cpu"
    report = {
        "weights": Path(args.weights).name,
        "data": args.data,
        "split": args.split,
        "images": len(list(Path(check_det_dataset(args.data)[args.split]).glob("*.*"))),
        "imgsz": args.imgsz,
        "overall": {
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50-95": round(float(m.box.map), 4),
        },
        "per_class": per_class,
        "speed_ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
        "hardware": torch.cuda.get_device_name(0) if on_gpu else platform.processor() or "cpu",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
    }
    out = Path(args.out or ROOT / "results" / f"metrics_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
