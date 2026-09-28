"""`make loop DOMAIN=…`: eval the champion → one optimiser session → eval the challenger → gate → ledger.

Each cycle is one line in `loop/<domain>/ledger.jsonl`, written before the
challenger runs and completed after the gate. The optimiser never sees the
gate's verdict except through that ledger on the next cycle, which is the
point: the record of what worked is a file it reads, not a memory it keeps.
The challenger then runs the test split once, promoted or held, and the
champion's test run on the same tasks is compared with it (`test_compare`) —
for the ledger and the tables. That comparison never decides anything: the
test split is reported, never optimised on, and a gate that read it would be
optimising on it one step removed.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from tau2_loop.agent.versions import load_version
from tau2_loop.data.splits import split_ids
from tau2_loop.eval.compare import compare
from tau2_loop.eval.results import TaskResult
from tau2_loop.eval.runner import RunMeta, list_runs, load_run, run_eval
from tau2_loop.loop.ledger import append_entry, next_cycle_number, update_entry
from tau2_loop.loop.optimiser import run_optimiser
from tau2_loop.tracking.registry import promote, read_registry, register

console = Console()


def _covers(meta: RunMeta, ids: list[str], trials: int) -> bool:
    """A run scored exactly these tasks at this many trials (a run on an older cut does not)."""
    return set(meta.task_ids) == set(ids) and meta.trials == trials and not meta.dry_run


def _champion_run(
    domain: str, agent: str, concurrency: int, trials: int = 1
) -> tuple[RunMeta, list[TaskResult]]:
    """Reuse the registry's train run when it is the same bytes on the split's current tasks;
    otherwise re-evaluate, and make that run the champion's record."""
    reg = read_registry(domain)
    champ = reg.get("champion") or {}
    version = load_version(domain, agent)
    same_bytes = champ.get("agent") == agent and champ.get("fingerprint") == version.fingerprint
    if same_bytes and champ.get("split") == "train":
        try:
            meta, results = load_run(str(champ["run_id"]))
            if _covers(meta, split_ids(domain, "train"), trials):
                return meta, results
        except FileNotFoundError:
            pass
    meta, results = run_eval(
        domain,
        agent,
        split="train",
        trials=trials,
        concurrency=concurrency,
        note="champion re-eval for loop",
    )
    if not champ:
        promote(meta.run_id)
    elif same_bytes:
        promote(meta.run_id, kind="re-baseline")
    return meta, results


def _test_run(
    domain: str, agent: str, ids: list[str], trials: int, concurrency: int, note: str
) -> tuple[RunMeta, list[TaskResult]]:
    """The newest test run of these bytes on exactly these tasks, or a new one."""
    version = load_version(domain, agent)
    for m in reversed(list_runs(domain)):
        if (
            m.agent == agent
            and m.fingerprint == version.fingerprint
            and m.split == "test"
            and m.summary
            and _covers(m, ids, trials)
        ):
            return load_run(m.run_id)
    return run_eval(domain, agent, split="test", trials=trials, concurrency=concurrency, note=note)


async def run_cycle(
    domain: str,
    agent: str,
    optimiser_model: str,
    concurrency: int,
    run_test: bool = True,
    trials: int = 1,
) -> dict[str, Any]:
    cycle = next_cycle_number(domain)
    champion = load_version(domain, agent)
    champ_meta, champ_results = _champion_run(domain, agent, concurrency, trials)
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
        "trials": trials,
        "split_version": champ_meta.split_version,
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
        trials=trials,
        concurrency=concurrency,
        note=f"loop cycle {cycle} challenger",
    )
    verdict = compare(champ_results, chall_results)
    still_failed = sorted(
        {r.task_id for r in chall_results if r.correct is False and r.task_id in failed_ids}
    )
    outcome: dict[str, Any] = {
        "verdict": "promote" if verdict.promote else "hold",
        "reason": verdict.reason,
        "rule": verdict.rule,
        "p_value": verdict.p_value,
        "passes": f"{verdict.champion_passed} → {verdict.challenger_passed}",
        "pass_1": f"{verdict.champion_pass1:.3f} → {verdict.challenger_pass1:.3f}",
        "pass_k": {
            "champion": (champ_meta.summary or {}).get("pass_hat_k"),
            "challenger": (chall_meta.summary or {}).get("pass_hat_k"),
        },
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
    else:
        console.print(f"[yellow]hold on {agent}[/]: {verdict.reason} · {outcome['passes']}")
    # recorded before the test runs, so a cycle that dies there still carries its verdict
    update_entry(domain, cycle, outcome=outcome, tokens={**opt_tokens, **eval_tokens})
    if run_test:
        test_meta, test_results = run_eval(
            domain,
            opt.new_version,
            split="test",
            trials=trials,
            concurrency=concurrency,
            note=f"test split, cycle {cycle} ({outcome['verdict']})",
        )
        ts = test_meta.summary or {}
        outcome["test_run"] = test_meta.run_id
        outcome["test_passes"] = f"{ts.get('passed')}/{ts.get('n_scored')}"
        champ_test_meta, champ_test = _test_run(
            domain,
            agent,
            list(test_meta.task_ids),
            trials,
            concurrency,
            note=f"champion's test run for cycle {cycle}'s comparison",
        )
        try:
            tv = compare(champ_test, test_results)
        except ValueError as e:  # a champion test run on other tasks: say so, keep the cycle
            outcome["test_compare"] = {"champion_run": champ_test_meta.run_id, "error": str(e)}
        else:
            # reported beside the verdict, never used by it
            outcome["test_compare"] = {
                "champion_run": champ_test_meta.run_id,
                "challenger_run": test_meta.run_id,
                "passes": f"{tv.champion_passed} → {tv.challenger_passed}",
                "pass_1": f"{tv.champion_pass1:.3f} → {tv.challenger_pass1:.3f}",
                "fixed": tv.fixed,
                "broken": tv.broken,
                "p_value": tv.p_value,
                "reason": tv.reason,
            }
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
    trials: int = 1,
) -> None:
    current = agent or (read_registry(domain).get("champion") or {}).get("agent") or "v0"
    for _ in range(cycles):
        entry = await run_cycle(domain, current, optimiser_model, concurrency, trials=trials)
        outcome = entry.get("outcome") or {}
        if outcome.get("verdict") == "promote":
            current = str(entry["challenger"])
        if outcome.get("verdict") == "nothing to fix":
            break
