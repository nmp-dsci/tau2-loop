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

`make challenge DOMAIN=… AGENT=vN` scores a version nobody's optimiser wrote — a
model-swap fork, say — against the champion through the same code: everything a
cycle does once its challenger exists is `_score_challenger`, called by both.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from rich.console import Console

from tau2_loop.agent.versions import load_version
from tau2_loop.data.splits import split_ids
from tau2_loop.eval.compare import compare
from tau2_loop.eval.results import TaskResult
from tau2_loop.eval.runner import RunMeta, list_runs, load_run, run_eval
from tau2_loop.llm import model_label
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


def _runs_of(domain: str, agent: str, split: str, ids: list[str], trials: int) -> list[RunMeta]:
    """Runs of this version's exact bytes (name and fingerprint) on these tasks, newest first."""
    version = load_version(domain, agent)
    return [
        m
        for m in reversed(list_runs(domain))
        if m.agent == agent
        and m.fingerprint == version.fingerprint
        and m.split == split
        and _covers(m, ids, trials)
    ]


def _reuse_or_run(
    domain: str, agent: str, split: str, ids: list[str], trials: int, concurrency: int, note: str
) -> tuple[RunMeta, list[TaskResult]]:
    """The newest scored run of these bytes on exactly these tasks, or a new one."""
    for m in _runs_of(domain, agent, split, ids, trials):
        if m.summary:
            return load_run(m.run_id)
    return run_eval(domain, agent, split=split, trials=trials, concurrency=concurrency, note=note)


def _test_run(
    domain: str, agent: str, ids: list[str], trials: int, concurrency: int, note: str
) -> tuple[RunMeta, list[TaskResult]]:
    """The newest test run of these bytes on exactly these tasks, or a new one."""
    return _reuse_or_run(domain, agent, "test", ids, trials, concurrency, note)


# A run folder with no summary is being written by another process, or died. One that
# started this recently is taken to be running: a challenge waits for it rather than
# paying for the same conversations twice.
IN_FLIGHT_HOURS = 12


def _refuse_in_flight(domain: str, agent: str, split: str, ids: list[str], trials: int) -> None:
    runs = _runs_of(domain, agent, split, ids, trials)
    if any(m.summary for m in runs):
        return  # a scored run exists, and it is the one reused
    now = datetime.now(UTC)
    for m in runs:
        try:
            started = datetime.fromisoformat(m.started_at)
        except ValueError:
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        if now - started < timedelta(hours=IN_FLIGHT_HOURS):
            raise RuntimeError(
                f"runs/{m.run_id} ({agent} on {split}, started {m.started_at}) has no summary "
                "yet: another process is still producing it. Wait for it to finish and run the "
                f"challenge again; it reuses that run. A run that died stops blocking "
                f"{IN_FLIGHT_HOURS} h after it started."
            )


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

    return _score_challenger(
        entry,
        champ_results,
        champ_meta,
        concurrency=concurrency,
        trials=trials,
        run_test=run_test,
        reuse=False,
        base_tokens=opt_tokens,
    )


def _score_challenger(
    entry: dict[str, Any],
    champ_results: list[TaskResult],
    champ_meta: RunMeta,
    *,
    concurrency: int,
    trials: int,
    run_test: bool,
    reuse: bool,
    base_tokens: dict[str, Any],
) -> dict[str, Any]:
    """Everything a cycle does once its challenger exists, for a loop cycle and a challenge alike.

    `entry` is already in the ledger as `pending`. The challenger's train run → the gate
    against the champion's → register, promote or hold → the outcome in the ledger → its test
    run and the champion's on the same tasks → `test_compare` → MLflow. `reuse` takes the
    newest scored run of the challenger's exact bytes on each split instead of a new one (a
    hand-made version may have been scored already); a loop's challenger is bytes nobody has
    run, so a cycle passes False and always evaluates it.
    """
    domain = str(entry["domain"])
    cycle = int(entry["cycle"])
    agent = str(entry["champion"])
    challenger = str(entry["challenger"])
    failed_ids = list(entry["failed"])
    note = f"loop cycle {cycle} challenger"
    if reuse:
        ids = split_ids(domain, "train")
        chall_meta, chall_results = _reuse_or_run(
            domain, challenger, "train", ids, trials, concurrency, note
        )
    else:
        chall_meta, chall_results = run_eval(
            domain, challenger, split="train", trials=trials, concurrency=concurrency, note=note
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
    tokens = {**base_tokens, **eval_tokens}
    register(chall_meta.run_id, "challenger")
    if verdict.promote:
        promote(chall_meta.run_id)
        console.print(
            f"[green]promoted {challenger}[/]: {outcome['passes']} · fixed {verdict.fixed}"
        )
    else:
        console.print(f"[yellow]hold on {agent}[/]: {verdict.reason} · {outcome['passes']}")
    # recorded before the test runs, so a cycle that dies there still carries its verdict
    update_entry(domain, cycle, outcome=outcome, tokens=tokens)
    if run_test:
        test_note = f"test split, cycle {cycle} ({outcome['verdict']})"
        if reuse:
            test_meta, test_results = _test_run(
                domain, challenger, split_ids(domain, "test"), trials, concurrency, test_note
            )
        else:
            test_meta, test_results = run_eval(
                domain,
                challenger,
                split="test",
                trials=trials,
                concurrency=concurrency,
                note=test_note,
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
    update_entry(domain, cycle, outcome=outcome, tokens=tokens)
    try:
        from tau2_loop.tracking.mlflow_log import log_cycle

        log_cycle({**entry, "outcome": outcome})
    except Exception:  # noqa: BLE001
        pass
    entry["outcome"] = outcome
    return entry


def _diagnosis(domain: str, agent: str) -> dict[str, Any]:
    """The version's own `diagnosis.json` (a fork's says `kind`, `forked_from`, `agent_yaml`)."""
    p = load_version(domain, agent).path / "diagnosis.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def run_challenge(
    domain: str,
    challenger: str,
    concurrency: int = 3,
    trials: int = 1,
    run_test: bool = True,
) -> dict[str, Any]:
    """Score a version no optimiser wrote — a model-swap fork, say — against the champion.

    The same gate, ledger entry, promotion and test report as a loop cycle (`_score_challenger`),
    with no optimiser session. The champion's train run is `_champion_run`'s; the challenger's
    train and test runs are the newest scored runs of its exact bytes when they exist, else
    new ones. Refuses the champion itself, or a version with the champion's bytes.
    """
    champ_name = (read_registry(domain).get("champion") or {}).get("agent")
    if not champ_name:
        raise ValueError(f"{domain} has no champion: promote a run first (make promote RUN=…)")
    champ_name = str(champ_name)
    if challenger == champ_name:
        raise ValueError(f"{challenger} is {domain}'s champion: a challenge needs another version")
    champion = load_version(domain, champ_name)
    version = load_version(domain, challenger)
    if version.fingerprint == champion.fingerprint:
        raise ValueError(
            f"{challenger} has {champ_name}'s bytes (fingerprint {version.fingerprint}): "
            "the gate would compare the champion with itself"
        )
    _refuse_in_flight(domain, challenger, "train", split_ids(domain, "train"), trials)
    if run_test:
        _refuse_in_flight(domain, challenger, "test", split_ids(domain, "test"), trials)

    cycle = next_cycle_number(domain)
    champ_meta, champ_results = _champion_run(domain, champ_name, concurrency, trials)
    failures = [r for r in champ_results if r.correct is False or r.error]
    diag = _diagnosis(domain, challenger)
    kind = str(diag.get("kind") or "challenge")
    console.rule(
        f"[bold]{domain} · cycle {cycle}[/] · {kind} · {champ_name} → {challenger} "
        f"({model_label(version.config.model)}, {version.config.effort}) · no optimiser"
    )
    console.print(f"champion's train run: runs/{champ_meta.run_id} (the registry's)")
    newer = [
        m.run_id
        for m in _runs_of(domain, champ_name, "train", split_ids(domain, "train"), trials)
        if m.summary and m.run_id > champ_meta.run_id
    ]
    if newer:
        console.print(
            f"[yellow]note:[/] runs/{newer[0]} is a newer train run of {champ_name}'s bytes; the "
            f"gate reads the registry's. `make promote RUN={newer[0]} KIND=re-baseline` makes "
            "it the champion's record for the next challenge."
        )
    entry: dict[str, Any] = {
        "cycle": cycle,
        "domain": domain,
        "kind": kind,
        "champion": champ_name,
        "champion_run": champ_meta.run_id,
        "challenger": challenger,
        "forked_from": diag.get("forked_from"),
        "agent_yaml": diag.get("agent_yaml", []),
        "challenger_model": model_label(version.config.model),
        "challenger_effort": version.config.effort,
        "optimiser_model": None,
        "trials": trials,
        "split_version": champ_meta.split_version,
        "failed": [r.task_id for r in failures],
        # nobody diagnosed anything: the ledger's history reads these as strings and lists
        "diagnoses": [],
        "prompt_diff_summary": str(diag.get("prompt_diff_summary") or ""),
        "helper_diff_summary": str(diag.get("helper_diff_summary") or ""),
        "expected_to_fix": [],
        "risks": diag.get("risks", []),
        "optimiser": None,
        "tokens": {},
        "outcome": {"verdict": "pending"},
    }
    append_entry(domain, entry)
    return _score_challenger(
        entry,
        champ_results,
        champ_meta,
        concurrency=concurrency,
        trials=trials,
        run_test=run_test,
        reuse=True,
        base_tokens={},
    )


async def run_loop(
    domain: str,
    cycles: int = 1,
    agent: str | None = None,
    optimiser_model: str = "opus",
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
