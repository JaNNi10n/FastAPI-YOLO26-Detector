"""Fine-tune a YOLO detector on a public dataset and copy the best weights into models/.

Default run (what produced models/yolo26n-african-wildlife.pt):

    python scripts/train.py --model yolo26n.pt --data african-wildlife.yaml --epochs 30

Ultralytics downloads the base weights and the dataset on first use. Training
curves and the run config are copied into results/train/ so the numbers in the
README can be traced back to a file.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo26n.pt", help="base checkpoint to fine-tune")
    p.add_argument("--data", default="african-wildlife.yaml", help="Ultralytics dataset yaml")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default=str(ROOT / "runs"), help="where Ultralytics writes the run")
    p.add_argument("--name", default="yolo26n-african-wildlife")
    p.add_argument("--out", default=str(ROOT / "models" / "yolo26n-african-wildlife.pt"))
    p.add_argument(
        "--resume",
        metavar="LAST_PT",
        help="continue an interrupted run from its weights/last.pt (keeps the run's args; --batch/--device override)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from ultralytics import YOLO

    start = time.perf_counter()
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True, batch=args.batch, device=args.device, workers=args.workers)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            project=args.project,
            name=args.name,
            exist_ok=True,
            plots=True,
        )
    minutes = (time.perf_counter() - start) / 60

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)

    # Keep the evidence (curves + config) in the repo, not just in runs/ (git-ignored).
    res_dir = ROOT / "results" / "train"
    res_dir.mkdir(parents=True, exist_ok=True)
    for f in ("results.csv", "args.yaml", "results.png", "confusion_matrix_normalized.png"):
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, res_dir / f)

    summary = {
        "base_model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_this_invocation": args.batch,
        "seed": args.seed,
        "train_minutes_this_invocation": round(minutes, 1),
        "resumed_from": args.resume,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != "cpu" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "weights": out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out),
    }
    (res_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
