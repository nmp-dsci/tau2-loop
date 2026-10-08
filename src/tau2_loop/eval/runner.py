"""`make eval`: run a version over a split through tau2, score it, write `runs/<id>/`.

A run folder is the unit everything else reads — the tracker logs it, the gate
compares two of them, the optimiser reads its traces, the viewer lists them.
tau2's own results file (every message of every conversation, with
`reward_info`) is kept verbatim as `tau2_results.json`; `results.jsonl` is our
one-row-per-conversation view of it and `traces/` one file per conversation.
Nothing is kept only in MLflow.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from tau2_loop.agent.versions import AgentVersion, load_version
from tau2_loop.config import (
    DOMAINS,
    ROOT,
    RUNS_DIR,
    SMOKE_DOMAIN,
    SPLIT_SEED,
    quiet_tau2,
    settings,
)
from tau2_loop.data.splits import split_ids, split_version
from tau2_loop.eval.results import (
    Summary,
    TaskResult,
    from_simulation,
    read_results,
    slug,
    summarise,
    write_results,
)
from tau2_loop.llm import EFFORT, HARNESS, redact_tree, resolve_model, sdk_model

console = Console()

# the customer every run uses unless `run_eval(user_model=)` says otherwise (s16). Sonnet since
# s18 (7 Oct 2026, the person's call): Haiku's customers wrote calls out as text in 88 of 665
# banking conversations; Sonnet's few failures (as Claude, a given tool denied) are what the
# customer check sends back. Haiku before; the loop never reuses a run with another customer.
USER_MODEL = "sonnet"
USER_EFFORT = EFFORT
# The simulation's rules beyond tau2's own: our customer (`eval.user`) holds a stop sent with words
# until the agent's turn, and neither side may take the real date the Claude CLI tells every
# session (the customer is told the world's time, the agent `compose.CLOCK_NOTE`). Bump it when
# they change; the loop compares only runs that share it. /2 (s18, 7 Oct 2026): a customer text
# turn that speaks as an AI, writes a call out as text, denies a given tool or quotes an unseen id
# is sent back (`user.check_customer`).
SIM_RULES = "tau2_loop/2"
JUDGE_MODEL = "haiku"
JUDGE_EFFORT = EFFORT

# A held service call can sleep through a subscription window's reset (up to
# core.MAX_WAIT_S), so the HTTP client waits longer than that before giving up.
SERVICE_TIMEOUT_S = 7 * 3600

TAU2_SHA_FALLBACK = "2174a60"


def _tau2_sha() -> str:
    """The submodule's actual pinned commit; the fallback covers the demo image, which has no git."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT / "vendor" / "tau2-bench"), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or TAU2_SHA_FALLBACK
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return TAU2_SHA_FALLBACK


@dataclass
class RunMeta:
    run_id: str
    domain: str
    agent: str  # version name, e.g. v0
    fingerprint: str
    model: str
    user_model: str
    judge_model: str
    split: str
    n_tasks: int
    trials: int
    concurrency: int
    seed: int
    started_at: str
    finished_at: str | None = None
    code_sha: str = "unknown"
    tau2_sha: str = field(default_factory=_tau2_sha)
    summary: dict[str, Any] | None = None
    mlflow_run_id: str | None = None
    note: str = ""
    task_ids: list[str] = field(default_factory=list)
    tool_mode: str = "json"
    sampling: str = "cli-default"  # the SDK exposes no temperature; tau2's 0.0 does not apply
    dry_run: bool = False
    harness: str = (
        "baseline"  # llm.HARNESS at run time; "baseline" = inherited connector tools + title call
    )
    # Added with split v2; older run.json files read these defaults.
    agent_effort: str | None = None
    user_effort: str | None = None
    split_version: int | None = (
        None  # 1 = the 20 / 20 cut, 2 = half of base each, 3 = banking's 60 / 37
    )
    agent_route: str = "in-process"  # or service:<host:port>, the agent's calls over HTTP
    # banking only: tau2's retrieval variant (s09); None on other domains and on older runs,
    # which all ran `bm25`
    retrieval: str | None = None
    # the simulation's rules beyond tau2's (`SIM_RULES`); None = every run before 2 October 2026:
    # tau2's own customer, which ended a conversation on any stop, and both sides shown the real date
    sim_rules: str | None = None
    # a run joined from two (`extend_run`): the scored run and the run of the tasks its split had
    # gained since, same bytes and rules; None for a run played whole
    composed_of: list[str] | None = None
    # native only: every tool call of a reply kept (s14 P0a); False on every run before it
    parallel_calls: bool = False


def agent_route() -> tuple[str, str | None]:
    """Where the task agent's model calls go: ('in-process', None) or ('service:<host>', base url)."""
    url = settings().agent_service_url
    if not url:
        return "in-process", None
    host = url.split("://", 1)[-1].split("/", 1)[0]
    return f"service:{host}", url


def new_run_id(domain: str, version: AgentVersion, split: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{domain}_{version.name}_{split}"


def _run_config(
    domain: str,
    version: AgentVersion,
    ids: list[str],
    trials: int,
    concurrency: int,
    seed: int,
    user_model: str = USER_MODEL,
) -> Any:
    from tau2.data_model.simulation import TextRunConfig

    from tau2_loop.agent.factory import AGENT_NAME, SERVICE_PREFIX, VERSION_KEY, with_effort
    from tau2_loop.eval import user

    # tau2 writes these into tau2_results.json: provenance only, never a secret
    # (the service's bearer token is added by the factory at run time).
    llm_agent = sdk_model(version.config.model)
    llm_args_agent: dict[str, Any] = with_effort({VERSION_KEY: version.ref}, version.config.effort)
    _, url = agent_route()
    if url:
        llm_agent = SERVICE_PREFIX + resolve_model(version.config.model)
        llm_args_agent |= {"api_base": f"{url}/v1", "timeout": SERVICE_TIMEOUT_S}
    kwargs: dict[str, Any] = {
        "domain": domain,
        "task_set_name": domain,
        "task_split_name": None if domain == SMOKE_DOMAIN else "base",
        "task_ids": ids,
        "agent": AGENT_NAME,
        "llm_agent": llm_agent,
        "llm_args_agent": llm_args_agent,
        "user": user.register(domain),
        "llm_user": sdk_model(user_model),
        "llm_args_user": with_effort({}, USER_EFFORT),
        "num_trials": trials,
        "max_concurrency": concurrency,
        "seed": seed,
        "max_steps": version.config.max_steps,
        "log_level": "ERROR",
        "max_retries": 1,
    }
    if domain == "banking_knowledge":
        from tau2_loop.eval.retrieval import register as register_retrieval

        register_retrieval()  # the local AllTools variants a version may name
        kwargs["retrieval_config"] = version.retrieval
    return TextRunConfig(**kwargs)


def _point_judge_at_sdk() -> None:
    """tau2's NL judge names its model as a module constant; route it to the subscription too."""
    import tau2.evaluator.evaluator_nl_assertions as nl

    from tau2_loop.agent.factory import with_effort

    nl.DEFAULT_LLM_NL_ASSERTIONS = sdk_model(JUDGE_MODEL)
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = with_effort({}, JUDGE_EFFORT)


def run_eval(
    domain: str,
    agent: str = "v0",
    split: str = "train",
    trials: int = 1,
    concurrency: int = 3,
    task_ids: list[str] | None = None,
    note: str = "",
    track: bool = True,
    dry_run: bool = False,
    seed: int = SPLIT_SEED,
    user_model: str = USER_MODEL,
) -> tuple[RunMeta, list[TaskResult]]:
    if domain not in DOMAINS and domain != SMOKE_DOMAIN:
        raise ValueError(f"domain must be one of {DOMAINS + (SMOKE_DOMAIN,)}, got {domain!r}")
    quiet_tau2()
    if not dry_run:
        from tau2_loop.llm import require_live

        require_live()  # before a run folder exists: a refused run leaves nothing behind
    version = load_version(domain, agent)
    ids = task_ids or split_ids(domain, split)
    run_id = new_run_id(domain, version, split if not task_ids else "custom")
    run_dir = RUNS_DIR / run_id
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    meta = RunMeta(
        run_id=run_id,
        domain=domain,
        agent=version.name,
        fingerprint=version.fingerprint,
        model=sdk_model(version.config.model),
        user_model=sdk_model(user_model),
        judge_model=sdk_model(JUDGE_MODEL),
        split=split,
        n_tasks=len(ids),
        trials=trials,
        concurrency=concurrency,
        seed=seed,
        started_at=datetime.now(UTC).isoformat(),
        code_sha=settings().code_sha,
        note=note,
        task_ids=ids,
        tool_mode=version.config.tool_mode,
        dry_run=dry_run,
        harness=HARNESS,
        agent_effort=version.config.effort,
        user_effort=USER_EFFORT,
        split_version=None if task_ids else split_version(domain),
        agent_route=agent_route()[0],
        retrieval=version.retrieval,
        sim_rules=None if dry_run else SIM_RULES,
        parallel_calls=version.config.parallel_calls,
    )
    _write_meta(run_dir, meta)
    (run_dir / "agent").mkdir(exist_ok=True)
    for name, text in version.files().items():
        (run_dir / "agent" / name).write_text(text)

    console.rule(
        f"[bold]{run_id}[/] · {len(ids)} tasks × {trials} trial(s) · {meta.model} "
        f"({meta.agent_effort}, {meta.agent_route}) · concurrency={concurrency}"
    )
    if dry_run:
        results = [
            TaskResult(task_id=t, trial=i + 1, reward=0.0, correct=False, error="dry-run")
            for t in ids
            for i in range(trials)
        ]
        return _finish(run_dir, meta, results, track=False)

    from tau2_loop.agent.factory import register

    register()
    _point_judge_at_sdk()

    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks
    from tau2.runner.helpers import get_tasks

    config = _run_config(domain, version, ids, trials, concurrency, seed, user_model)
    tasks = get_tasks(domain, config.task_split_name, task_ids=ids)
    by_id = {t.id: t for t in tasks}
    tau2_results = run_tasks(
        config,
        tasks,
        save_path=run_dir / "tau2_results.json",
        evaluation_type=EvaluationType.ALL,
        console_display=False,
    )
    results = []
    for sim in tau2_results.simulations:
        trial = int(sim.trial or 0) + 1
        name = f"{slug(sim.task_id)}.json" if trials == 1 else f"{slug(sim.task_id)}_t{trial}.json"
        (run_dir / "traces" / name).write_text(
            json.dumps(sim.model_dump(mode="json"), ensure_ascii=False, indent=1)
        )
        row = from_simulation(sim, by_id.get(sim.task_id), name)
        row.trial = trial
        results.append(row)
    order = {t: i for i, t in enumerate(ids)}
    results.sort(key=lambda r: (order.get(r.task_id, 1 << 30), r.trial))
    for r in results:
        mark = "[green]pass[/]" if r.correct else "[red]FAIL[/]"
        console.print(
            f"  {r.task_id[:48]:<48} t{r.trial} {mark}  reward={r.reward:.2f}  "
            f"{r.n_agent_turns} agent turns · {r.n_tool_calls} tool calls · {r.termination_reason}"
            + (f"  [red]{r.error}[/]" if r.error else "")
        )
    return _finish(run_dir, meta, results, track=track)


def _finish(
    run_dir: Path, meta: RunMeta, results: list[TaskResult], track: bool
) -> tuple[RunMeta, list[TaskResult]]:
    write_results(run_dir / "results.jsonl", results)
    redact_tree(run_dir)  # belt to the provider's braces: nothing under runs/ names the account
    summary = summarise(results)
    meta.finished_at = datetime.now(UTC).isoformat()
    meta.summary = asdict(summary)
    _write_meta(run_dir, meta)
    _print_summary(summary)
    if track and not meta.dry_run:
        try:
            from tau2_loop.tracking.mlflow_log import log_run

            meta.mlflow_run_id = log_run(run_dir, meta, results)
            _write_meta(run_dir, meta)
            from tau2_loop.tracking.tracing import flush, log_run_traces

            n = log_run_traces(meta, results, run_dir)
            flush()
            console.print(f"[dim]mlflow: {n} conversation traces[/]")
        except Exception as e:  # noqa: BLE001 - tracking down never fails an eval
            console.print(f"[yellow]mlflow: not logged ({type(e).__name__}: {e})[/]")
    return meta, results


def extend_run(
    base_run_id: str, concurrency: int = 3, note: str = ""
) -> tuple[RunMeta, list[TaskResult]]:
    """Bring a scored run up to its split's current tasks without replaying what it scored: play
    only the tasks the split has gained (a custom run of them), then join the two into one run of
    the whole split (`compose_runs`). Refused unless the version's bytes, the simulation rules
    and the retrieval are the run's, and every task it scored is still on its side."""
    base, _ = load_run(base_run_id)
    if base.dry_run or not base.summary:
        raise ValueError(f"{base_run_id} is not a scored run")
    version = load_version(base.domain, base.agent)
    if version.fingerprint != base.fingerprint:
        raise ValueError(
            f"{base.agent}'s bytes changed since {base_run_id} ({base.fingerprint} → "
            f"{version.fingerprint}): extending it would join two versions"
        )
    if base.sim_rules != SIM_RULES:
        raise ValueError(f"{base_run_id} ran under other simulation rules ({base.sim_rules})")
    if base.user_model != sdk_model(USER_MODEL):
        raise ValueError(f"{base_run_id} ran with another customer ({base.user_model})")
    ids = split_ids(base.domain, base.split)
    moved = sorted(set(base.task_ids) - set(ids))
    if moved:
        raise ValueError(f"{base_run_id} scored tasks no longer on its {base.split} side: {moved}")
    missing = [t for t in ids if t not in set(base.task_ids)]
    if not missing:
        raise ValueError(f"{base_run_id} already covers {base.domain}'s {base.split} split")
    ext, _ = run_eval(
        base.domain,
        base.agent,
        base.split,
        base.trials,
        concurrency,
        task_ids=missing,
        note=note
        or f"the {len(missing)} {base.split} tasks split v{split_version(base.domain)} added "
        f"to {base_run_id}",
    )
    return compose_runs(base_run_id, ext.run_id)


def compose_runs(base_run_id: str, ext_run_id: str) -> tuple[RunMeta, list[TaskResult]]:
    """One run of a split joined from two of the same version: their conversations, results and
    tau2 results side by side in a new folder that names both (`composed_of`). The two must be
    the same bytes, model, customer, rules, retrieval and trials, on disjoint tasks that together
    are exactly the split. Neither source folder changes."""
    base, base_rows = load_run(base_run_id)
    ext, ext_rows = load_run(ext_run_id)
    same = (
        "domain",
        "agent",
        "fingerprint",
        "model",
        "user_model",
        "split",
        "trials",
        "agent_effort",
        "user_effort",
        "retrieval",
        "sim_rules",
    )
    differ = [k for k in same if getattr(base, k) != getattr(ext, k)]
    if differ:
        raise ValueError(f"{base_run_id} and {ext_run_id} differ in {differ}")
    if not (base.summary and ext.summary) or base.dry_run or ext.dry_run:
        raise ValueError("both runs must be scored")
    ids = split_ids(base.domain, base.split)
    both = set(base.task_ids) & set(ext.task_ids)
    if both or set(base.task_ids) | set(ext.task_ids) != set(ids):
        raise ValueError(
            f"the two runs must split {base.domain}'s {base.split} tasks between them "
            f"(shared {sorted(both)})"
        )
    version = load_version(base.domain, base.agent)
    run_id = new_run_id(base.domain, version, base.split)
    run_dir = RUNS_DIR / run_id
    (run_dir / "traces").mkdir(parents=True)
    for src in (base_run_id, ext_run_id):
        for f in sorted((RUNS_DIR / src / "traces").iterdir()):
            (run_dir / "traces" / f.name).write_bytes(f.read_bytes())
    (run_dir / "agent").mkdir()
    for f in sorted((RUNS_DIR / base_run_id / "agent").iterdir()):
        (run_dir / "agent" / f.name).write_bytes(f.read_bytes())
    tau2 = [
        json.loads((RUNS_DIR / r / "tau2_results.json").read_text())
        for r in (base_run_id, ext_run_id)
    ]
    joined = {
        **tau2[0],
        "tasks": tau2[0]["tasks"] + tau2[1]["tasks"],
        "simulations": tau2[0]["simulations"] + tau2[1]["simulations"],
        "simulation_index": None,
    }
    (run_dir / "tau2_results.json").write_text(json.dumps(joined, ensure_ascii=False))
    order = {t: i for i, t in enumerate(ids)}
    rows = sorted(base_rows + ext_rows, key=lambda r: (order[r.task_id], r.trial))
    write_results(run_dir / "results.jsonl", rows)
    summary = summarise(rows)
    meta = RunMeta(
        **{
            **asdict(base),
            "run_id": run_id,
            "n_tasks": len(ids),
            "task_ids": ids,
            "split_version": split_version(base.domain),
            "finished_at": ext.finished_at,
            "code_sha": base.code_sha
            if base.code_sha == ext.code_sha
            else f"{base.code_sha}+{ext.code_sha}",
            "summary": asdict(summary),
            "mlflow_run_id": None,  # each part is logged as it was played
            "note": f"{base_run_id} ({len(base.task_ids)} tasks) joined with {ext_run_id} "
            f"({len(ext.task_ids)} tasks): split v{split_version(base.domain)}",
            "composed_of": [base_run_id, ext_run_id],
        }
    )
    _write_meta(run_dir, meta)
    console.rule(f"[bold]{run_id}[/] · {meta.note}")
    _print_summary(summary)
    return meta, rows


def _print_summary(s: Summary) -> None:
    console.print(
        f"[bold]{s.passed}/{s.n_scored} pass[/] ({(s.pass_rate or 0):.0%})  {s.pass_hat_k}  "
        f"terminations={s.by_termination}  errors={s.errored_ids}"
    )


def _write_meta(run_dir: Path, meta: RunMeta) -> None:
    (run_dir / "run.json").write_text(json.dumps(asdict(meta), indent=2) + "\n")


def load_run(run_id: str) -> tuple[RunMeta, list[TaskResult]]:
    run_dir = RUNS_DIR / run_id
    if not (run_dir / "run.json").exists():
        raise FileNotFoundError(f"no run at {run_dir}")
    meta = RunMeta(**json.loads((run_dir / "run.json").read_text()))
    results = (
        read_results(run_dir / "results.jsonl") if (run_dir / "results.jsonl").exists() else []
    )
    return meta, results


def list_runs(domain: str | None = None) -> list[RunMeta]:
    metas: list[RunMeta] = []
    if not RUNS_DIR.exists():
        return metas
    for d in sorted(RUNS_DIR.iterdir()):
        if (d / "run.json").exists():
            m = RunMeta(**json.loads((d / "run.json").read_text()))
            if domain is None or m.domain == domain:
                metas.append(m)
    return metas
