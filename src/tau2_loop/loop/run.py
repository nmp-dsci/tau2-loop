"""`make loop DOMAIN=…`: eval the champion → one optimiser session → eval the challenger → gate → ledger.

Each cycle is one line in `loop/<domain>/ledger.jsonl`, written before the
challenger runs and completed after the gate. The optimiser never sees the
gate's verdict except through that ledger on the next cycle, which is the
point: the record of what worked is a file it reads, not a memory it keeps.
Promotion runs the test split once, for the ledger and the README table, and
never feeds back into the gate.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from tau2_loop.agent.versions import load_version
from tau2_loop.eval.compare import compare
from tau2_loop.eval.results import TaskResult
from tau2_loop.eval.runner import RunMeta, load_run, run_eval
from tau2_loop.loop.ledger import append_entry, next_cycle_number, update_entry
from tau2_loop.loop.optimiser import run_optimiser
from tau2_loop.tracking.registry import promote, read_registry, register

console = Console()


def _champion_run(domain: str, agent: str, concurrency: int) -> tuple[RunMeta, list[TaskResult]]:
    """Reuse the registry's train run when it is the same bytes; otherwise re-evaluate."""
    reg = read_registry(domain)
    champ = reg.get("champion") or {}
    version = load_version(domain, agent)
    if (
        champ.get("agent") == agent
        and champ.get("fingerprint") == version.fingerprint
        and champ.get("split") == "train"
    ):
        try:
            return load_run(str(champ["run_id"]))
        except FileNotFoundError:
            pass
    meta, results = run_eval(
        domain, agent, split="train", concurrency=concurrency, note="champion re-eval for loop"
    )
    if not champ:
        promote(meta.run_id)
    return meta, results


async def run_cycle(
    domain: str, agent: str, optimiser_model: str, concurrency: int, test_after_promote: bool = True
) -> dict[str, Any]:
    cycle = next_cycle_number(domain)
    champion = load_version(domain, agent)
    champ_meta, champ_results = _champion_run(domain, agent, concurrency)
    failures = [r for r in champ_results if r.correct is False or r.error]
    console.rule(f"[bold]{domain} · cycle {cycle}[/] · champion {agent} · {len(failures)} failures")
    if not failures:
        clean: dict[str, Any] = {
            "cycle": cycle,
            "domain": domain,
            "champion": agent,
            "champion_run": champ_meta.run_id,
            "challenger": None,
            "failed": [],
            "outcome": {"verdict": "nothing to fix"},
        }
        append_entry(domain, clean)
        return clean

    opt = await run_optimiser(champion, champ_meta.run_id, failures, model=optimiser_model)
    failed_ids = [r.task_id for r in failures]
    opt_tokens = {"optimiser_in": opt.input_tokens, "optimiser_out": opt.output_tokens}
    entry: dict[str, Any] = {
        "cycle": cycle,
        "domain": domain,
        "champion": agent,
        "champion_run": champ_meta.run_id,
        "challenger": opt.new_version,
        "optimiser_model": optimiser_model,
        "failed": failed_ids,
        "diagnoses": opt.diagnosis.get("diagnoses", []),
        "prompt_diff_summary": opt.diagnosis.get("prompt_diff_summary", ""),
        "helper_diff_summary": opt.diagnosis.get("helper_diff_summary", ""),
        "expected_to_fix": opt.diagnosis.get("expected_to_fix", []),
        "risks": opt.diagnosis.get("risks", []),
        "optimiser": {
            "turns": opt.n_turns,
            "duration_ms": opt.duration_ms,
            "cost_usd_est": opt.cost_usd,
            "error": opt.error,
        },
        "tokens": opt_tokens,
        "outcome": {"verdict": "pending"},
    }
    append_entry(domain, entry)
    if opt.error and (
        "outside agents" in opt.error or "frozen" in opt.error or "no change" in opt.error
    ):
        update_entry(domain, cycle, outcome={"verdict": "rejected", "reason": opt.error})
        console.print(f"[red]cycle {cycle} rejected:[/] {opt.error}")
        entry["outcome"] = {"verdict": "rejected", "reason": opt.error}
        return entry

    chall_meta, chall_results = run_eval(
        domain,
        opt.new_version,
        split="train",
        concurrency=concurrency,
        note=f"loop cycle {cycle} challenger",
    )
    verdict = compare(champ_results, chall_results)
    still_failed = [
        r.task_id for r in chall_results if r.correct is False and r.task_id in failed_ids
    ]
    outcome: dict[str, Any] = {
        "verdict": "promote" if verdict.promote else "hold",
        "reason": verdict.reason,
        "rule": verdict.rule,
        "p_value": verdict.p_value,
        "passes": f"{verdict.champion_passed} → {verdict.challenger_passed}",
        "fixed": verdict.fixed,
        "broken": verdict.broken,
        "still_failed": still_failed,
        "challenger_run": chall_meta.run_id,
    }
    eval_tokens = {
        "eval_agent_in": sum(r.agent_input_tokens for r in chall_results),
        "eval_agent_out": sum(r.agent_output_tokens for r in chall_results),
        "eval_user_in": sum(r.user_input_tokens for r in chall_results),
        "eval_user_out": sum(r.user_output_tokens for r in chall_results),
    }
    register(chall_meta.run_id, "challenger")
    if verdict.promote:
        promote(chall_meta.run_id)
        console.print(
            f"[green]promoted {opt.new_version}[/]: {outcome['passes']} · fixed {verdict.fixed}"
        )
        if test_after_promote:
            test_meta, test_results = run_eval(
                domain,
                opt.new_version,
                split="test",
                concurrency=concurrency,
                note=f"test split after promotion in cycle {cycle}",
            )
            ts = test_meta.summary or {}
            outcome["test_run"] = test_meta.run_id
            outcome["test_passes"] = f"{ts.get('passed')}/{ts.get('n_scored')}"
    else:
        console.print(f"[yellow]hold on {agent}[/]: {verdict.reason} · {outcome['passes']}")
    update_entry(domain, cycle, outcome=outcome, tokens={**opt_tokens, **eval_tokens})
    try:
        from tau2_loop.tracking.mlflow_log import log_cycle

        log_cycle({**entry, "outcome": outcome})
    except Exception:  # noqa: BLE001
        pass
    entry["outcome"] = outcome
    return entry


async def run_loop(
    domain: str,
    cycles: int = 1,
    agent: str | None = None,
    optimiser_model: str = "sonnet",
    concurrency: int = 3,
) -> None:
    current = agent or (read_registry(domain).get("champion") or {}).get("agent") or "v0"
    for _ in range(cycles):
        entry = await run_cycle(domain, current, optimiser_model, concurrency)
        outcome = entry.get("outcome") or {}
        if outcome.get("verdict") == "promote":
            current = str(entry["challenger"])
        if outcome.get("verdict") == "nothing to fix":
            break
