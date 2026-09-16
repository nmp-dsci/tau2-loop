#!/usr/bin/env bash
# Smoke-test a demo URL. Asserts what makes this deployment what it claims to
# be: up, in demo mode, serving the committed evidence, and holding no route
# that could call a model. A deploy failing any of these fails the workflow.
#
#   ./scripts/demo_smoke.sh https://xyz.ap-southeast-1.awsapprunner.com
set -euo pipefail

BASE="${1:?usage: demo_smoke.sh <base-url>}"
BASE="${BASE%/}"
fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

# 1. Health says demo, and at least one domain has a champion.
health="$(curl -fsS --max-time 20 "$BASE/healthz")" || fail "health unreachable"
echo "$health" | grep -q '"mode":"demo"' || fail "mode is not demo: $health"
echo "$health" | grep -q '"airline":"v' || fail "no airline champion registered: $health"

# 2. The committed evidence is served: runs, the ledgers, the four domains with 40 tasks each.
n_runs="$(curl -fsS --max-time 20 "$BASE/api/runs" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
[[ "$n_runs" -gt 0 ]] || fail "no runs served"
n_ledger="$(curl -fsS --max-time 20 "$BASE/api/ledger" | python3 -c 'import json,sys; print(sum(len(v) for v in json.load(sys.stdin).values()))')"
[[ "$n_ledger" -gt 0 ]] || fail "ledgers are empty"
for d in airline retail telecom banking_knowledge; do
  n_tasks="$(curl -fsS --max-time 20 "$BASE/api/domains/$d" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["tasks"]))')"
  [[ "$n_tasks" -eq 40 ]] || fail "expected 40 tasks for $d, got $n_tasks"
done

# 3. There is no ask route: a POST to it is a 404/405, never a model call.
code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 -X POST "$BASE/api/ask" -H 'Content-Type: application/json' -d '{"question":"x"}')"
[[ "$code" == "404" || "$code" == "405" ]] || fail "unexpected /api/ask status $code"

# 4. The React shell serves from the same origin, for a deep link too.
curl -fsS --max-time 20 "$BASE/" | grep -qi '<!doctype html' || fail "index did not render"
curl -fsS --max-time 20 "$BASE/loop" | grep -qi '<!doctype html' || fail "deep link did not render"

echo "SMOKE PASS: $BASE — mode=demo, $n_runs runs, $n_ledger ledger entries, 4 domains × 40 tasks, no model route"
