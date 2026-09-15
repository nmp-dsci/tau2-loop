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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from tau2_loop.agent.versions import AgentVersion, load_version
from tau2_loop.config import DOMAINS, RUNS_DIR, SMOKE_DOMAIN, SPLIT_SEED, quiet_tau2, settings
from tau2_loop.data.splits import split_ids
from tau2_loop.eval.results import (
    Summary,
    TaskResult,
    from_simulation,
    read_results,
    slug,
    summarise,
    write_results,
)
from tau2_loop.llm import HARNESS, redact_tree, sdk_model

console = Console()

USER_MODEL = "haiku"
JUDGE_MODEL = "haiku"


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
    tau2_sha: str = "2174a60"
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


def new_run_id(domain: str, version: AgentVersion, split: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{domain}_{version.name}_{split}"


def _run_config(
    domain: str, version: AgentVersion, ids: list[str], trials: int, concurrency: int, seed: int
) -> Any:
    from tau2.data_model.simulation import TextRunConfig

    from tau2_loop.agent.factory import AGENT_NAME, VERSION_KEY

    kwargs: dict[str, Any] = {
        "domain": domain,
        "task_set_name": domain,
        "task_split_name": None if domain == SMOKE_DOMAIN else "base",
        "task_ids": ids,
        "agent": AGENT_NAME,
        "llm_agent": sdk_model(version.config.model),
        "llm_args_agent": {VERSION_KEY: version.ref},
        "user": "user_simulator",
        "llm_user": sdk_model(USER_MODEL),
        "llm_args_user": {},
        "num_trials": trials,
        "max_concurrency": concurrency,
        "seed": seed,
        "max_steps": version.config.max_steps,
        "log_level": "ERROR",
        "max_retries": 1,
    }
    if domain == "banking_knowledge":
        kwargs["retrieval_config"] = "bm25"
    return TextRunConfig(**kwargs)


def _point_judge_at_sdk() -> None:
    """tau2's NL judge names its model as a module constant; route it to the subscription too."""
    import tau2.evaluator.evaluator_nl_assertions as nl

    nl.DEFAULT_LLM_NL_ASSERTIONS = sdk_model(JUDGE_MODEL)
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {}


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
        user_model=sdk_model(USER_MODEL),
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
    )
    _write_meta(run_dir, meta)
    (run_dir / "agent").mkdir(exist_ok=True)
    for name, text in version.files().items():
        (run_dir / "agent" / name).write_text(text)

    console.rule(
        f"[bold]{run_id}[/] · {len(ids)} tasks × {trials} trial(s) · {meta.model} · concurrency={concurrency}"
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

    config = _run_config(domain, version, ids, trials, concurrency, seed)
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
        except Exception as e:  # noqa: BLE001 - tracking down never fails an eval
            console.print(f"[yellow]mlflow: not logged ({type(e).__name__}: {e})[/]")
    return meta, results


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
