"""The CI gate: every committed champion is what its registry says it is.

Runs with no model and no server. For each domain with a champion it checks
that the run folder's pass count matches the registry, that the agent folder's
bytes still hash to the fingerprint the run was scored under, that no ledger
cycle is left `pending`, and — the expensive one — that replaying the saved
trajectories through tau2's evaluators (`eval/rescore.py`) reproduces the
recorded pass/fail on every conversation. A prompt edit that forgets to
re-run, or a hand edit to a run folder, fails here.
"""

from __future__ import annotations

import sys

from tau2_loop.agent.versions import load_version
from tau2_loop.config import DOMAINS
from tau2_loop.eval.runner import load_run
from tau2_loop.loop.ledger import read_ledger
from tau2_loop.tracking.registry import read_registry


def check(rescore: bool = True) -> list[str]:
    problems: list[str] = []
    any_champion = False
    for domain in DOMAINS:
        reg = read_registry(domain)
        champ = reg.get("champion")
        if not champ:
            continue
        any_champion = True
        try:
            meta, results = load_run(str(champ["run_id"]))
        except FileNotFoundError as e:
            problems.append(f"{domain}: {e}")
            continue
        passed = sum(1 for r in results if r.correct)
        if passed != champ["passed"]:
            problems.append(
                f"{domain}: champion {champ['agent']} results.jsonl has {passed} passes, registry says {champ['passed']}"
            )
        version = load_version(domain, str(champ["agent"]))
        if version.fingerprint != champ["fingerprint"]:
            problems.append(
                f"{domain}: agents/{domain}/{champ['agent']} hashes to {version.fingerprint}, but the champion run was scored on {champ['fingerprint']}: re-run `make eval DOMAIN={domain}` and promote"
            )
        if meta.fingerprint != champ["fingerprint"]:
            problems.append(
                f"{domain}: run {meta.run_id} fingerprint {meta.fingerprint} != registry {champ['fingerprint']}"
            )
        for entry in read_ledger(domain):
            if (entry.get("outcome") or {}).get("verdict") == "pending":
                problems.append(f"{domain}: ledger cycle {entry.get('cycle')} is still pending")
        if rescore:
            from tau2_loop.eval.rescore import rescore_run

            try:
                table = rescore_run(meta.run_id)
            except Exception as e:  # noqa: BLE001
                problems.append(
                    f"{domain}: re-score of {meta.run_id} failed: {type(e).__name__}: {e}"
                )
                continue
            disagree = [k for k, v in table.items() if not v["agree"]]
            if disagree:
                problems.append(
                    f"{domain}: re-scoring {meta.run_id} disagrees with the recorded verdict on {disagree[:5]}"
                )
    if not any_champion:
        problems.append("no champion registered in any loop/<domain>/registry.json")
    return problems


def main() -> int:
    quick = "--no-rescore" in sys.argv
    problems = check(rescore=not quick)
    if problems:
        for p in problems:
            print(f"GATE FAIL: {p}", file=sys.stderr)
        return 1
    for domain in DOMAINS:
        c = read_registry(domain).get("champion")
        if c:
            print(
                f"GATE OK: {domain} champion {c['agent']} ({c['fingerprint']}) {c['passed']}/{c['n_scored']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
