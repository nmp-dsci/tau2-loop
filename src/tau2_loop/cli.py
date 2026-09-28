"""`tau2loop` — the command line behind every Makefile target."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def splits() -> None:
    """Cut each domain's base set in half, train / test (seed 300; the 20 / 20 cut kept inside it)."""
    from tau2_loop.data.splits import read_split, write_splits

    for p in write_splits():
        s = read_split(p.stem)
        console.print(
            f"{p}: v{s['version']} · base {s['base_n']} → train {len(s['train'])} · "
            f"test {len(s['test'])} · reserve {s['reserve_n']}"
        )


@app.command()
def fork(
    domain: str = "airline",
    model: str | None = None,
    effort: str | None = None,
    source: Annotated[str | None, typer.Option("--from")] = None,
) -> None:
    """A new version with the champion's prompt and helper and a different model or effort."""
    from tau2_loop.agent.versions import fork_version
    from tau2_loop.tracking.registry import champion_name

    src = source or champion_name(domain) or "v0"
    v = fork_version(domain, src, model=model, effort=effort)
    console.print(
        f"agents/{domain}/{v.name}: {src}'s surfaces · model {v.config.model} · "
        f"effort {v.config.effort} · fingerprint {v.fingerprint}"
    )


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
def promote(run_id: str, kind: str = "gate") -> None:
    """Make a run's version the champion of its domain (--kind "model swap" for a fork by fiat)."""
    from tau2_loop.tracking.registry import promote as _promote

    console.print(_promote(run_id, kind=kind))


@app.command()
def loop(
    domain: str = "airline",
    cycles: int = 1,
    agent: str | None = None,
    optimiser: str = "sonnet",
    concurrency: int = 3,
    trials: int = 1,
) -> None:
    """The error loop on one domain: eval → diagnose → new version → gate → test → ledger."""
    from tau2_loop.loop.run import run_loop

    asyncio.run(run_loop(domain, cycles, agent, optimiser, concurrency, trials))


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
def serve(port: int = 8080, host: str = "127.0.0.1", reload: bool = False) -> None:
    """Run the API (and the built frontend when frontend/dist exists).

    `--reload` restarts it when the package's code changes, as Vite does for the
    frontend; without it a long-lived `make dev` serves routes older than the page.
    """
    import uvicorn

    uvicorn.run(
        "tau2_loop.serving.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        # the package only: a loop writing runs/ must not restart the server mid-request
        reload_dirs=[str(Path(__file__).parent)] if reload else None,
    )


@app.command()
def smoke(
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
    concurrency: int = 2,
    agent: str = "v0",
) -> None:
    """The adapter's smoke test: a mock-domain version (10 tasks), all three roles on the subscription."""
    from tau2_loop.eval.runner import run_eval

    meta, _ = run_eval("mock", agent, "all", 1, concurrency, task, note="adapter smoke on mock")
    console.print(f"run: runs/{meta.run_id}")


@app.command("agent-service")
def agent_service(port: int = 8090, docker: bool = False, build: bool = True) -> None:
    """The task agent's model call as a service on 127.0.0.1:PORT (POST /v1/chat/completions).

    `--docker` builds `Dockerfile.agent` and runs it, handing the container only
    CLAUDE_CODE_OAUTH_TOKEN and AGENT_SERVICE_TOKEN (from the environment, `.env` or
    `~/.env`); without it the service runs here, on this machine's CLI login. Point
    a run at it with AGENT_SERVICE_URL=http://127.0.0.1:PORT.
    """
    import os
    import secrets
    import subprocess

    from dotenv import dotenv_values

    from tau2_loop.config import ROOT

    names = ("AGENT_SERVICE_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    found: dict[str, str] = {}
    for src in (
        dict(os.environ),
        dotenv_values(ROOT / ".env"),
        dotenv_values(Path.home() / ".env"),
    ):
        for n in names:
            if n not in found and (src.get(n) or "").strip():
                found[n] = str(src[n]).strip()
    if "AGENT_SERVICE_TOKEN" not in found:
        found["AGENT_SERVICE_TOKEN"] = secrets.token_urlsafe(32)
        with (ROOT / ".env").open("a") as f:
            f.write(f"AGENT_SERVICE_TOKEN={found['AGENT_SERVICE_TOKEN']}\n")
        console.print("wrote a new AGENT_SERVICE_TOKEN to .env (gitignored)")
    if not docker:
        import uvicorn

        os.environ["AGENT_SERVICE_TOKEN"] = found["AGENT_SERVICE_TOKEN"]
        uvicorn.run("tau2_loop.llm.service:create_app", factory=True, host="127.0.0.1", port=port)
        return
    if "CLAUDE_CODE_OAUTH_TOKEN" not in found:
        console.print(
            "[red]CLAUDE_CODE_OAUTH_TOKEN is not set: run `claude setup-token` and put it in ~/.env[/]"
        )
        raise typer.Exit(1)
    image = "tau2loop-agent"
    if build:
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(ROOT / "Dockerfile.agent"),
                "-t",
                image,
                str(ROOT / "src/tau2_loop/llm"),
            ],
            check=True,
        )
    # the values travel in the child's environment, never on a command line `ps` can read
    env = {**os.environ, **found}
    cmd = ["docker", "run", "--rm", "--name", "tau2loop-agent", "-p", f"127.0.0.1:{port}:8080"]
    for n in names:
        cmd += ["-e", n]
    subprocess.run([*cmd, image], env=env, check=True)


if __name__ == "__main__":
    app()
