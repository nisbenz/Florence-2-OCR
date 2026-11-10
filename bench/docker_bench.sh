#!/usr/bin/env bash
# One experiment row for journal.md: start the image under the official limits, run accuracy,
# latency and load, report peak container memory, stop the container.
#
#   bench/docker_bench.sh <samples> <references> [CPUS] [-e VAR=value ...]
#   bench/docker_bench.sh ~/data/samples ~/data/references 4 -e ORT_THREADS=4
set -euo pipefail
SAMPLES=$1 REFERENCES=$2 CPUS=${3:-4}
shift $(( $# < 3 ? $# : 3 ))
PY=${PY:-python}
IMAGE=${IMAGE:-florence-ocr}
cd "$(dirname "$0")/.."

docker run -d --name ocr-bench --rm -p 8000:8000 --memory=4g --memory-swap=4g --cpus="$CPUS" "$@" "$IMAGE" >/dev/null
trap 'docker stop ocr-bench >/dev/null 2>&1 || true' EXIT
start=$(date +%s)
curl -s --retry 120 --retry-delay 1 --retry-all-errors --fail localhost:8000/health >/dev/null
echo "healthy after $(( $(date +%s) - start )) s (cpus=$CPUS $*)"

( while docker ps -q --filter name=ocr-bench | grep -q .; do
    docker stats --no-stream --format '{{.MemUsage}}' ocr-bench 2>/dev/null | cut -d/ -f1
  done ) > /tmp/ocr-bench-mem.log &

"$PY" bench/accuracy.py --samples "$SAMPLES" --references "$REFERENCES" | tail -1
"$PY" bench/latency.py --samples "$SAMPLES" --rounds "${ROUNDS:-3}"
"$PY" bench/load.py --samples "$SAMPLES" --steps ${STEPS:-1 2 4 8} --duration "${DURATION:-30}"
echo "peak memory: $(sort -h /tmp/ocr-bench-mem.log | tail -1)"
docker ps -q --filter name=ocr-bench | grep -q . || echo "CONTAINER DIED (OOM?)"
