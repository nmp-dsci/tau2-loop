#!/usr/bin/env bash
# v0 on the train split of every scored domain, one after another; logs under workspace/.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p workspace
for d in ${DOMAINS:-airline retail telecom banking_knowledge}; do
  uv run tau2loop eval --domain "$d" --agent v0 --split train --concurrency "${CONCURRENCY:-3}" > "workspace/baseline_$d.log" 2>&1 || echo "FAILED $d" >> workspace/baselines.status
  echo "done $d $(date -u +%H:%M:%S)" >> workspace/baselines.status
done
