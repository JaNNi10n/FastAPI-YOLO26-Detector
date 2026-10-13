#!/usr/bin/env bash
# Reproduce everything in results/ and docs/ from scratch.
#   bash scripts/reproduce.sh            # train + evaluate + test + benchmark + demo
#   SKIP_TRAIN=1 bash scripts/reproduce.sh   # reuse models/yolo26n-african-wildlife.pt
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PORT="${PORT:-8000}"

if [[ -z "${SKIP_TRAIN:-}" ]]; then
  "$PY" scripts/train.py --model yolo26n.pt --data african-wildlife.yaml --epochs 30
fi

"$PY" scripts/evaluate.py --split val
"$PY" scripts/evaluate.py --split test
"$PY" -m ruff check . && "$PY" -m ruff format --check .
"$PY" -m pytest -q | tee results/pytest.txt

# Latency: one uvicorn worker per device, 100 sequential requests each.
for dev in 0 cpu; do
  label=$([[ "$dev" == cpu ]] && echo cpu || echo gpu)
  DEVICE=$dev "$PY" -m uvicorn app.main:app --port "$PORT" --log-level warning &
  pid=$!
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 1; done
  "$PY" scripts/benchmark_api.py --label "$label" --n 100 --url "http://127.0.0.1:$PORT"
  if [[ "$label" == gpu ]]; then
    curl -s -F "file=@tests/assets/elephant_zebra.jpg;type=image/jpeg" \
      "http://127.0.0.1:$PORT/predict" | "$PY" -m json.tool > docs/demo_response.json
    curl -s -F "file=@tests/assets/elephant_zebra.jpg;type=image/jpeg" \
      "http://127.0.0.1:$PORT/predict/annotated" -o docs/demo_annotated.jpg
  fi
  kill "$pid"; wait "$pid" 2>/dev/null || true
done
