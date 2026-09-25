"""`tau2loop` — the command line behind every Makefile target."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def splits() -> None:
    """Cut the 20 / 20 train / test split per domain (seed 300) and extract the selected tasks."""
    from tau2_loop.data.splits import read_split, write_splits

    for p in write_splits():
        s = read_split(p.stem)
        console.print(f"{p}: base {s['base_n']} → train {len(s['train'])} · test {len(s['test'])}")


@app.command()
def eval(  # noqa: A001 - the Makefile target is `eval`
    domain: str = "airline",
    agent: str = "v0",
    split: str = "train",
    trials: int = 1,
    concurrency: int = 3,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
    note: str = "",
    no_track: bool = False,
    dry_run: bool = False,
) -> None:
    """Run an agent version over a split of one domain and score it."""
    from tau2_loop.eval.runner import run_eval

    meta, _ = run_eval(
        domain, agent, split, trials, concurrency, task, note, track=not no_track, dry_run=dry_run
    )
    console.print(f"run: runs/{meta.run_id}")


@app.command()
def score(run_id: str) -> None:
    """Summarise a run folder from its results.jsonl."""
    from tau2_loop.eval.results import summarise
    from tau2_loop.eval.runner import load_run

    _, results = load_run(run_id)
    console.print(summarise(results))


@app.command()
def rescore(run_id: str) -> None:
    """Replay a run's saved trajectories through tau2's evaluators, offline, and compare verdicts."""
    from tau2_loop.eval.rescore import rescore_run

    table = rescore_run(run_id)
    t = Table("conversation", "recorded", "rescored", "components", "agree")
    for k, v in table.items():
        t.add_row(
            k[:60], str(v["recorded"]), str(v["rescored"]), str(v["components"]), str(v["agree"])
        )
    console.print(t)
    bad = [k for k, v in table.items() if not v["agree"]]
    console.print(
        f"{len(table) - len(bad)}/{len(table)} agree" + (f" · disagree: {bad}" if bad else "")
    )


@app.command()
def compare(champion: str, challenger: str) -> None:
    """Gate two runs of the same tasks: one-sided exact McNemar."""
    from tau2_loop.eval.compare import compare as _compare
    from tau2_loop.eval.runner import load_run

    _, a = load_run(champion)
    _, b = load_run(challenger)
    v = _compare(a, b)
    console.print(v)


@app.command()
def register(run_id: str, alias: str = "challenger") -> None:
    """Register a run's version under an alias in its domain's registry."""
    from tau2_loop.tracking.registry import register as _register

    console.print(_register(run_id, alias))


@app.command()
def promote(run_id: str) -> None:
    """Make a run's version the champion of its domain."""
    from tau2_loop.tracking.registry import promote as _promote

    console.print(_promote(run_id))


@app.command()
def loop(
    domain: str = "airline",
    cycles: int = 1,
    agent: str | None = None,
    optimiser: str = "sonnet",
    concurrency: int = 3,
) -> None:
    """The error loop on one domain: eval → diagnose → new version → gate → ledger."""
    from tau2_loop.loop.run import run_loop

    asyncio.run(run_loop(domain, cycles, agent, optimiser, concurrency))


@app.command()
def ledger(domain: str = "airline") -> None:
    """Print a domain's loop ledger."""
    from tau2_loop.loop.ledger import read_ledger

    t = Table("cycle", "champion", "challenger", "verdict", "passes", "fixed", "broken", "p")
    for e in read_ledger(domain):
        o = e.get("outcome") or {}
        t.add_row(
            str(e.get("cycle")),
            str(e.get("champion")),
            str(e.get("challenger")),
            str(o.get("verdict")),
            str(o.get("passes", "")),
            str(o.get("fixed", "")),
            str(o.get("broken", "")),
            f"{o['p_value']:.3f}" if "p_value" in o else "",
        )
    console.print(t)


@app.command()
def leaderboard() -> None:
    """Ingest τ²-bench's published submissions into data/index/leaderboard.json."""
    from tau2_loop.data.leaderboard import best_per_domain, ingest

    payload = ingest()
    console.print(
        f"[bold]{len(payload['entries'])}[/] of {payload['listed']} listed submissions "
        f"from tau2-bench@{payload['tau2_sha']}"
    )
    for domain, best in best_per_domain(payload["entries"]).items():
        console.print(f"  {domain:20} best pass^1 {best['pass_1']:.1f}  {best['model']}")


@app.command()
def snapshot() -> None:
    """Export the MLflow experiment to loop/mlflow_snapshot.json for the demo image."""
    from tau2_loop.tracking.snapshot import write_snapshot

    console.print(write_snapshot())


@app.command()
def gate(no_rescore: bool = False) -> None:
    """The CI gate: every champion re-scores to what its registry says."""
    import sys

    from tau2_loop.tracking.gate import check

    problems = check(rescore=not no_rescore)
    for p in problems:
        console.print(f"[red]GATE FAIL:[/] {p}")
    if problems:
        sys.exit(1)
    console.print("[green]GATE OK[/]")


@app.command()
def serve(port: int = 8080, host: str = "127.0.0.1") -> None:
    """Run the API (and the built frontend when frontend/dist exists)."""
    import uvicorn

    uvicorn.run("tau2_loop.serving.app:create_app", factory=True, host=host, port=port)


@app.command()
def smoke(
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None, concurrency: int = 2
) -> None:
    """The adapter's smoke test: v0 on the mock domain (10 tasks), all three roles on the subscription."""
    from tau2_loop.eval.runner import run_eval

    meta, _ = run_eval("mock", "v0", "all", 1, concurrency, task, note="adapter smoke on mock")
    console.print(f"run: runs/{meta.run_id}")


if __name__ == "__main__":
    app()
