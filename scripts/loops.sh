#!/usr/bin/env bash
# M3: two cycles on airline, one on each other domain; logs under workspace/.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p workspace
rm -f workspace/loops.status
for spec in "airline 2" "retail 1" "telecom 1" "banking_knowledge 1"; do
  set -- $spec
  uv run tau2loop loop --domain "$1" --cycles "$2" --concurrency "${CONCURRENCY:-3}" > "workspace/loop_$1.log" 2>&1 || echo "FAILED $1" >> workspace/loops.status
  echo "done $1 $(date -u +%H:%M:%S)" >> workspace/loops.status
done
