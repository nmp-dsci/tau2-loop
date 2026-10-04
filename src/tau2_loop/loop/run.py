"""`make loop DOMAIN=…`: eval the champion → one optimiser session → eval the challenger → gate → ledger.

Each cycle is one line in `loop/<domain>/ledger.jsonl`, written before the
challenger runs and completed after the gate. The optimiser never sees the
gate's verdict except through that ledger on the next cycle, which is the
point: the record of what worked is a file it reads, not a memory it keeps.
The challenger then runs the test split once, promoted or held, and the
champion's test run on the same tasks is compared with it (`test_compare`) —
for the ledger and the tables. That comparison decides nothing: the test split
is reported, never optimised on, and a gate that read it would be optimising on
it one step removed. The exception is a domain in `GATE_ON_TEST` (banking, from
3 Oct 2026, the person's call): its optimiser reads every failure of all of
train, and its gate is that test comparison. No optimiser still sees a test
conversation or a test task's id.

`make challenge DOMAIN=… AGENT=vN` scores a version nobody's optimiser wrote — a
model-swap fork, say — against the champion through the same code: everything a
cycle does once its challenger exists is `_score_challenger`, called by both.

A domain whose train split is halved (s09 option B, `data/splits`) shows the
optimiser only its read half's failures and gates on the gate half, which no
optimiser sees; every version still runs all of train and test. No domain is
halved today: banking was, until its gate moved to test. `make ab` runs
two optimisers from one champion on the same input — the classic one and the
routing one — scores both, and crowns at most one (s09 §6).

Every cycle reads its domain's optimiser profile first (s13 §5,
`optimisers/<domain>/`, `loop/profiles.py`), so a profile edited between cycles
applies from the next one. `--optimiser` and `--mode` override its model and
mode for one run; the ledger entry keeps the profile's fingerprint and what was
overridden (`optimiser_profile`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from rich.console import Console

from tau2_loop.agent.versions import load_version, version_dir
from tau2_loop.config import GATE_ON_TEST
from tau2_loop.data.splits import halves, split_ids
from tau2_loop.eval.compare import compare
from tau2_loop.eval.results import TaskResult
from tau2_loop.eval.runner import SIM_RULES, RunMeta, list_runs, load_run, run_eval
from tau2_loop.llm import model_label
from tau2_loop.loop.ledger import append_entry, next_cycle_number, update_entry
from tau2_loop.loop.optimiser import MODES, OptimiserOutput, build_context, run_optimiser
from tau2_loop.loop.profiles import OptimiserProfile, load_profile
from tau2_loop.tracking.registry import promote, read_registry, register

console = Console()


def _covers(meta: RunMeta, ids: list[str], trials: int) -> bool:
    """A run scored exactly these tasks at this many trials, under today's simulation rules (a run
    on an older cut does not, nor one with tau2's own customer and the real date: `runner.SIM_RULES`)."""
    return (
        set(meta.task_ids) == set(ids)
        and meta.trials == trials
        and not meta.dry_run
        and meta.sim_rules == SIM_RULES
    )


def _read_failures(domain: str, results: list[TaskResult]) -> list[TaskResult]:
    """The champion's failures an optimiser may read: all of train, or its read half."""
    h = halves(domain)
    read = set(h[0]) if h else None
    return [
        r
        for r in results
        if (r.correct is False or r.error) and (read is None or r.task_id in read)
    ]


def _gate_rows(domain: str, results: list[TaskResult]) -> list[TaskResult]:
    """The rows the gate decides on: all of train, or its gate half."""
    h = halves(domain)
    if not h:
        return results
    gate = set(h[1])
    return [r for r in results if r.task_id in gate]


def _refuse_no_test(domain: str, run_test: bool) -> None:
    """A domain whose gate decides on test cannot be scored without its test run."""
    if domain in GATE_ON_TEST and not run_test:
        raise ValueError(
            f"{domain}'s gate decides on the test split: the test run cannot be skipped"
        )


def _gate_on(domain: str) -> str:
    if domain in GATE_ON_TEST:
        return f"test ({len(split_ids(domain, 'test'))} tasks)"
    h = halves(domain)
    return f"gate half ({len(h[1])} of {len(h[0]) + len(h[1])} train tasks)" if h else "train"


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


def _cycle_entry(
    cycle: int,
    domain: str,
    agent: str,
    champ_meta: RunMeta,
    opt: OptimiserOutput,
    optimiser_model: str,
    trials: int,
    failures: list[TaskResult],
    profile: OptimiserProfile,
    mode: str,
) -> dict[str, Any]:
    """A loop cycle's ledger line as it stands when its optimiser has finished. `optimiser_model`
    and `mode` are what it ran with; `optimiser_profile` the dataset's profile it ran under."""
    return {
        "cycle": cycle,
        "domain": domain,
        "champion": agent,
        "champion_run": champ_meta.run_id,
        "challenger": opt.new_version,
        "optimiser_model": optimiser_model,
        "optimiser_mode": opt.mode,
        "optimiser_profile": profile.ledger_record(optimiser_model, mode),
        "routed": opt.routed,
        "surfaces_changed": opt.surfaces_changed,
        "trials": trials,
        "split_version": champ_meta.split_version,
        "gate_on": _gate_on(domain),
        "failed": [r.task_id for r in failures],
        "diagnoses": opt.diagnosis.get("diagnoses", []),
        "prompt_diff_summary": opt.diagnosis.get("prompt_diff_summary", "") or "",
        "helper_diff_summary": opt.diagnosis.get("helper_diff_summary", "") or "",
        "expected_to_fix": opt.diagnosis.get("expected_to_fix", []),
        "risks": opt.diagnosis.get("risks", []),
        # what it took from the champion's earlier challengers, and what it left out (`champion_record`)
        "carried_forward": opt.diagnosis.get("carried_forward", []),
        "dropped": opt.diagnosis.get("dropped", []),
        "optimiser": {
            "turns": opt.n_turns,
            "duration_ms": opt.duration_ms,
            "cost_usd_est": opt.cost_usd,
            "error": opt.error,
        },
        "tokens": {"optimiser_in": opt.input_tokens, "optimiser_out": opt.output_tokens},
        "outcome": {"verdict": "pending"},
    }


def _rejected(opt: OptimiserOutput) -> bool:
    """A guard, a write outside the version, a frozen file or no change at all: no challenger runs."""
    legacy = ("outside agents", "frozen", "no change")
    return opt.rejected or bool(opt.error and any(x in opt.error for x in legacy))


def _reject(domain: str, entry: dict[str, Any], opt: OptimiserOutput) -> dict[str, Any]:
    outcome = {"verdict": "rejected", "reason": opt.error}
    update_entry(domain, int(entry["cycle"]), outcome=outcome)
    console.print(f"[red]cycle {entry['cycle']} rejected:[/] {opt.error}")
    entry["outcome"] = outcome
    return entry


def _settings(
    domain: str, optimiser_model: str | None, mode: str | None
) -> tuple[OptimiserProfile, str, str]:
    """The domain's profile, read now, and the model and mode a cycle runs: a flag's when given,
    else the profile's. A mode neither knows is refused before anything is evaluated."""
    profile = load_profile(domain)
    model = optimiser_model or profile.model
    mode = mode or profile.mode
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return profile, model, mode


async def run_cycle(
    domain: str,
    agent: str,
    optimiser_model: str | None = None,
    concurrency: int = 3,
    run_test: bool = True,
    trials: int = 1,
    mode: str | None = None,
) -> dict[str, Any]:
    """One cycle under the domain's optimiser profile, read afresh; `optimiser_model` and `mode`
    left None are the profile's."""
    _refuse_no_test(domain, run_test)
    profile, optimiser_model, mode = _settings(domain, optimiser_model, mode)
    cycle = next_cycle_number(domain)
    champion = load_version(domain, agent)
    champ_meta, champ_results = _champion_run(domain, agent, concurrency, trials)
    failures = _read_failures(domain, champ_results)
    console.rule(
        f"[bold]{domain} · cycle {cycle}[/] · champion {agent} · {len(failures)} failures to read"
        f" · {mode} optimiser ({optimiser_model}, profile {profile.fingerprint}"
        f"{'' if profile.is_default() else ', tuned'}) · gate on {_gate_on(domain)}"
    )
    if not failures:
        clean: dict[str, Any] = {
            "cycle": cycle,
            "domain": domain,
            "champion": agent,
            "champion_run": champ_meta.run_id,
            "challenger": None,
            "failed": [],
            "optimiser_profile": profile.ledger_record(optimiser_model, mode),
            "outcome": {"verdict": "nothing to fix"},
        }
        append_entry(domain, clean)
        return clean

    opt = await run_optimiser(
        champion, champ_meta.run_id, failures, model=optimiser_model, mode=mode, profile=profile
    )
    entry = _cycle_entry(
        cycle, domain, agent, champ_meta, opt, optimiser_model, trials, failures, profile, mode
    )
    append_entry(domain, entry)
    if _rejected(opt):
        return _reject(domain, entry, opt)
    return _score_challenger(
        entry,
        champ_results,
        champ_meta,
        concurrency=concurrency,
        trials=trials,
        run_test=run_test,
        reuse=False,
        base_tokens=dict(entry["tokens"]),
    )


async def run_ab(
    domain: str,
    optimiser_model: str | None = None,
    concurrency: int = 3,
    trials: int = 1,
    run_test: bool = True,
) -> list[dict[str, Any]]:
    """Two challengers from one champion, one per optimiser mode, on exactly the same input.

    Both optimisers read the same failures and the same history (one `Context`), and both run
    before either result reaches the ledger, so neither sees the other; the second cannot read
    the first's folder. Each challenger is gated against the champion and run on test. At most
    one is crowned: of those that pass the gate, the one with more gate-half passes, then fewer
    breaks; the other is recorded as held, with the reason. Both run under the domain's
    optimiser profile (its model unless `optimiser_model` is given); the pair is one per mode
    whatever mode the profile names."""
    _refuse_no_test(domain, run_test)
    profile, optimiser_model, _ = _settings(domain, optimiser_model, None)
    agent = str((read_registry(domain).get("champion") or {}).get("agent") or "")
    if not agent:
        raise ValueError(f"{domain} has no champion: promote a run first (make promote RUN=…)")
    champion = load_version(domain, agent)
    champ_meta, champ_results = _champion_run(domain, agent, concurrency, trials)
    failures = _read_failures(domain, champ_results)
    first = next_cycle_number(domain)
    console.rule(
        f"[bold]{domain} · A/B cycles {first}–{first + 1}[/] · champion {agent} · "
        f"{len(failures)} failures to read · gate on {_gate_on(domain)}"
    )
    if not failures:
        raise ValueError(f"{agent} has no failures to read on {domain}")
    ctx = build_context(champion, champ_meta.run_id, failures, profile)
    outs: list[OptimiserOutput] = []
    for mode in MODES:
        hidden = [version_dir(domain, o.new_version) for o in outs]
        console.print(f"[bold]{mode} optimiser[/] ({optimiser_model})")
        outs.append(
            await run_optimiser(
                champion,
                champ_meta.run_id,
                failures,
                model=optimiser_model,
                mode=mode,
                hidden=hidden,
                ctx=ctx,
                profile=profile,
            )
        )
    entries = []
    for i, (opt, mode) in enumerate(zip(outs, MODES, strict=True)):
        entry = _cycle_entry(
            first + i,
            domain,
            agent,
            champ_meta,
            opt,
            optimiser_model,
            trials,
            failures,
            profile,
            mode,
        )
        entry["experiment"] = {"name": "s09 A/B", "pair": [first, first + 1]}
        append_entry(domain, entry)
        entries.append(entry)
    scored: list[dict[str, Any]] = []
    for entry, opt in zip(entries, outs, strict=True):
        if _rejected(opt):
            scored.append(_reject(domain, entry, opt))
            continue
        scored.append(
            _score_challenger(
                entry,
                champ_results,
                champ_meta,
                concurrency=concurrency,
                trials=trials,
                run_test=run_test,
                reuse=False,
                base_tokens=dict(entry["tokens"]),
                crown=False,
            )
        )
    passed = [e for e in scored if (e.get("outcome") or {}).get("verdict") == "promote"]
    if passed:
        best = max(
            passed,
            key=lambda e: (
                e["outcome"]["gate_passes"]["challenger"],
                -len(e["outcome"].get("broken") or []),
            ),
        )
        promote(best["outcome"]["challenger_run"])
        console.print(f"[green]crowned {best['challenger']}[/] of the A/B pair")
        for e in passed:
            if e is not best:
                o = e["outcome"]
                o["verdict"] = "hold"
                o["reason"] = (
                    f"{o.get('reason')} · passed the gate; its A/B partner {best['challenger']} took the title"
                )
                update_entry(domain, int(e["cycle"]), outcome=o)
    return scored


def _test_pair(
    entry: dict[str, Any], why: str, trials: int, concurrency: int, reuse: bool
) -> tuple[RunMeta, list[TaskResult], RunMeta, list[TaskResult]]:
    """The challenger's test run and the champion's on the same tasks (the champion's newest run
    of its bytes on exactly those tasks, else a new one). `why` goes in the challenger run's note:
    the verdict when test is reported after it, the train run when the gate is on test."""
    domain, cycle = str(entry["domain"]), int(entry["cycle"])
    challenger, agent = str(entry["challenger"]), str(entry["champion"])
    note = f"test split, cycle {cycle} ({why})"
    if reuse:
        test_meta, test_results = _test_run(
            domain, challenger, split_ids(domain, "test"), trials, concurrency, note
        )
    else:
        test_meta, test_results = run_eval(
            domain, challenger, split="test", trials=trials, concurrency=concurrency, note=note
        )
    champ_test_meta, champ_test = _test_run(
        domain,
        agent,
        list(test_meta.task_ids),
        trials,
        concurrency,
        note=f"champion's test run for cycle {cycle}'s comparison",
    )
    return test_meta, test_results, champ_test_meta, champ_test


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
    crown: bool = True,
) -> dict[str, Any]:
    """Everything a cycle does once its challenger exists, for a loop cycle and a challenge alike.

    `entry` is already in the ledger as `pending`. The challenger's train run → the gate
    against the champion's → register, promote or hold → the outcome in the ledger → its test
    run and the champion's on the same tasks → `test_compare` → MLflow. `reuse` takes the
    newest scored run of the challenger's exact bytes on each split instead of a new one (a
    hand-made version may have been scored already); a loop's challenger is bytes nobody has
    run, so a cycle passes False and always evaluates it. The gate reads the gate half where
    train is halved, and the test comparison where the domain gates on test (`GATE_ON_TEST`),
    which then runs before the verdict and cannot be skipped. `crown=False` records a passing
    verdict without promoting it (an A/B pair crowns at most one, after both are scored).
    """
    domain = str(entry["domain"])
    cycle = int(entry["cycle"])
    agent = str(entry["champion"])
    challenger = str(entry["challenger"])
    failed_ids = list(entry["failed"])
    on_test = domain in GATE_ON_TEST
    _refuse_no_test(domain, run_test)
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
    test = _test_pair(entry, "the gate", trials, concurrency, reuse) if on_test else None
    if test:
        verdict = compare(test[3], test[1])
    else:
        verdict = compare(_gate_rows(domain, champ_results), _gate_rows(domain, chall_results))
    cs, hs = champ_meta.summary or {}, chall_meta.summary or {}
    h = halves(domain)
    # the fixes and breaks an optimiser may later read: the read half's where train is halved, all
    # of train's where the gate is on test. The gate's own ids stay in `fixed` / `broken` and
    # never reach a prompt.
    read = set(h[0]) if h else {r.task_id for r in champ_results} if on_test else None
    read_seen = (
        compare(
            [r for r in champ_results if r.task_id in read],
            [r for r in chall_results if r.task_id in read],
        )
        if read is not None
        else None
    )
    still_failed = sorted(
        {r.task_id for r in chall_results if r.correct is False and r.task_id in failed_ids}
    )
    outcome: dict[str, Any] = {
        "verdict": "promote" if verdict.promote else "hold",
        "reason": verdict.reason,
        "rule": verdict.rule,
        "p_value": verdict.p_value,
        "passes": f"{verdict.champion_passed} → {verdict.challenger_passed}",
        "gate_on": _gate_on(domain),
        "gate_passes": {
            "champion": verdict.champion_passed,
            "challenger": verdict.challenger_passed,
            "n": verdict.n,
        },
        "train_passes": f"{cs.get('passed')}/{cs.get('n_scored')} → {hs.get('passed')}/{hs.get('n_scored')}",
        "pass_1": f"{verdict.champion_pass1:.3f} → {verdict.challenger_pass1:.3f}",
        "pass_k": {
            "champion": (champ_meta.summary or {}).get("pass_hat_k"),
            "challenger": (chall_meta.summary or {}).get("pass_hat_k"),
        },
        "fixed": verdict.fixed,
        "broken": verdict.broken,
        "still_failed": still_failed,
        "challenger_run": chall_meta.run_id,
        **({"read_fixed": read_seen.fixed, "read_broken": read_seen.broken} if read_seen else {}),
    }
    eval_tokens = {
        "eval_agent_in": sum(r.agent_input_tokens for r in chall_results),
        "eval_agent_out": sum(r.agent_output_tokens for r in chall_results),
        "eval_user_in": sum(r.user_input_tokens for r in chall_results),
        "eval_user_out": sum(r.user_output_tokens for r in chall_results),
    }
    tokens = {**base_tokens, **eval_tokens}
    register(chall_meta.run_id, "challenger")
    if verdict.promote and crown:
        promote(chall_meta.run_id)
        console.print(
            f"[green]promoted {challenger}[/]: {outcome['passes']} · fixed {verdict.fixed}"
        )
    elif verdict.promote:
        console.print(f"[green]{challenger} passed the gate[/]: {outcome['passes']}; crowned later")
    else:
        console.print(f"[yellow]hold on {agent}[/]: {verdict.reason} · {outcome['passes']}")
    # recorded before the test runs, so a cycle that dies there still carries its verdict
    update_entry(domain, cycle, outcome=outcome, tokens=tokens)
    if run_test:
        test_meta, test_results, champ_test_meta, champ_test = test or _test_pair(
            entry, outcome["verdict"], trials, concurrency, reuse
        )
        ts = test_meta.summary or {}
        outcome["test_run"] = test_meta.run_id
        outcome["test_passes"] = f"{ts.get('passed')}/{ts.get('n_scored')}"
        try:
            tv = compare(champ_test, test_results)
        except ValueError as e:  # a champion test run on other tasks: say so, keep the cycle
            outcome["test_compare"] = {"champion_run": champ_test_meta.run_id, "error": str(e)}
        else:
            # the gate's own comparison where the domain gates on test; else reported beside the
            # verdict, never used by it
            outcome["test_compare"] = {
                "champion_run": champ_test_meta.run_id,
                "challenger_run": test_meta.run_id,
                "passes": f"{tv.champion_passed} → {tv.challenger_passed}",
                "pass_1": f"{tv.champion_pass1:.3f} → {tv.challenger_pass1:.3f}",
                "fixed": tv.fixed,
                "broken": tv.broken,
                "p_value": tv.p_value,
                "reason": tv.reason,
                "gated": on_test,
            }
    update_entry(domain, cycle, outcome=outcome, tokens=tokens)
    try:
        from tau2_loop.tracking.mlflow_log import log_cycle

        log_cycle({**entry, "outcome": outcome})
    except Exception:  # noqa: BLE001
        pass
    entry["outcome"] = outcome
    return entry


async def run_optimise(
    domain: str,
    source: str | None = None,
    optimiser_model: str | None = None,
    mode: str | None = None,
    settings: dict[str, Any] | None = None,
) -> OptimiserOutput:
    """One optimiser session on `source`'s train failures (default the champion), writing the
    next version and scoring nothing (s14): play it with `make eval`, gate it with `make
    challenge`, whose ledger entry then carries this diagnosis. `settings` gives the new version
    agent settings that differ from the source's (effort, model, parallel calls): the harness
    writes and freezes them before the session, so the optimiser still edits only surfaces.

    The source's failures come from the registry's train run when the source is the champion,
    else from the newest scored train run of its exact bytes; a source with neither is refused,
    because this step plays nothing."""
    profile, optimiser_model, mode = _settings(domain, optimiser_model, mode)
    champ_name = (read_registry(domain).get("champion") or {}).get("agent")
    source = source or (str(champ_name) if champ_name else None)
    if not source:
        raise ValueError(f"{domain} has no champion: name a source version (--from)")
    version = load_version(domain, source)
    meta: RunMeta | None = None
    results: list[TaskResult] = []
    reg = (read_registry(domain).get("champion") or {}) if source == champ_name else {}
    if reg.get("fingerprint") == version.fingerprint and reg.get("split") == "train":
        meta, results = load_run(str(reg["run_id"]))
    else:
        runs = _runs_of(domain, source, "train", split_ids(domain, "train"), 1)
        scored = [m for m in runs if m.summary]
        if scored:
            meta, results = load_run(scored[-1].run_id)
    if meta is None:
        raise ValueError(
            f"{domain}/{source} has no scored train run of its exact bytes: play one first "
            f"(make eval DOMAIN={domain} AGENT={source} SPLIT=train)"
        )
    failures = _read_failures(domain, results)
    if not failures:
        raise ValueError(f"{source} has no failures to read on {domain}'s train split")
    console.rule(
        f"[bold]{domain} · optimise[/] · from {source} (runs/{meta.run_id}) · {len(failures)} "
        f"failures to read · {mode} optimiser ({optimiser_model}, profile {profile.fingerprint}"
        f"{'' if profile.is_default() else ', tuned'})"
        + (f" · new agent settings {settings}" if settings else "")
    )
    opt = await run_optimiser(
        version,
        meta.run_id,
        failures,
        model=optimiser_model,
        mode=mode,
        profile=profile,
        settings=settings,
    )
    diag_file = version_dir(domain, opt.new_version) / "diagnosis.json"
    try:
        d = json.loads(diag_file.read_text())
    except (OSError, ValueError):
        d = None
    if isinstance(d, dict):  # what `make challenge` writes into the ledger for this version
        d["optimised_from"] = source
        d["source_run"] = meta.run_id
        d["optimiser"] = {
            "model": model_label(optimiser_model),
            "mode": mode,
            "profile": profile.ledger_record(optimiser_model, mode),
            "routed": opt.routed,
            "surfaces_changed": opt.surfaces_changed,
            "turns": opt.n_turns,
            "input_tokens": opt.input_tokens,
            "output_tokens": opt.output_tokens,
            "cost_usd_notional": opt.cost_usd,
            "duration_ms": opt.duration_ms,
            "error": opt.error,
            "rejected": opt.rejected,
        }
        diag_file.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    return opt


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
    _refuse_no_test(domain, run_test)
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
    failures = _read_failures(domain, champ_results)
    diag = _diagnosis(domain, challenger)
    optimised = isinstance(diag.get("optimiser"), dict)  # written by `make optimise` (s14)
    kind = str(diag.get("kind") or ("optimised" if optimised else "challenge"))
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
        "optimiser_model": (diag.get("optimiser") or {}).get("model") if optimised else None,
        "optimised_from": diag.get("optimised_from"),
        "trials": trials,
        "split_version": champ_meta.split_version,
        "gate_on": _gate_on(domain),
        "failed": [r.task_id for r in failures],
        # a fork: nobody diagnosed anything (the ledger's history reads these as strings and
        # lists); a `make optimise` version: its session's diagnosis
        "diagnoses": diag.get("diagnoses", []) if optimised else [],
        "prompt_diff_summary": str(diag.get("prompt_diff_summary") or ""),
        "helper_diff_summary": str(diag.get("helper_diff_summary") or ""),
        "expected_to_fix": diag.get("expected_to_fix", []) if optimised else [],
        "risks": diag.get("risks", []),
        "optimiser": diag.get("optimiser") if optimised else None,
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
    optimiser_model: str | None = None,
    concurrency: int = 3,
    trials: int = 1,
    mode: str | None = None,
) -> None:
    """`cycles` cycles from `agent` (default the champion), each under the domain's optimiser
    profile as it stands when the cycle starts; `optimiser_model` and `mode` override it for
    every cycle of this run."""
    current = agent or (read_registry(domain).get("champion") or {}).get("agent") or "v0"
    for _ in range(cycles):
        entry = await run_cycle(
            domain, current, optimiser_model, concurrency, trials=trials, mode=mode
        )
        outcome = entry.get("outcome") or {}
        if outcome.get("verdict") == "promote":
            current = str(entry["challenger"])
        if outcome.get("verdict") == "nothing to fix":
            break
