"""The merge bench (s30): one library job's merges replayed under a merge option, no research.

r19's re-merge (`make rag-remerge`) showed that a merge can be run again on the same research: each
session's conversation is rebuilt from its events, its draft and decision are reused, and only the
merge runs. The bench does that for one job and one option at a time, outside the rN line
(`merge_runs/`, immutable like `judge_runs/`): the source sessions' drafts are cut to the drafted
job that names the bench's job, the merges run in the build's lock order on a scratch library
seeded like the version they replay, and each reads the head the last one wrote.

  merge_bench/<domain>/bench.yaml        the jobs, their seed, sources and order; the grader
  merge_bench/<domain>/review.md         the reviewer every option shares (the derived standard)
  merge_bench/<domain>/<option>/         option.yaml (its `base` rN and settings) and prompts
  merge_runs/<stamp>_<domain>_<option>_t<trial>/
      run.json                           option, trial, job, status, tokens, prompt hashes
      rag_library/<domain>/bench_<opt>/  the scratch store: seed.json, versions/, commits.jsonl
      sessions/<session id>/             each merge's events.jsonl, output.json, score.json

A run stops at the first merge that fails (a usage limit included) and is never relaunched by the
bench: it records where it stopped. No gold reaches a merge; the eval (`merge_eval.py`) reads the
answer key, which no option's prompt does.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tau2_loop.config import ROOT
from tau2_loop.workflows import rag_agent, store

BENCH_DIR = ROOT / "merge_bench"
RUNS_DIR = ROOT / "merge_runs"
POLICY_FROM = "Write a discoverable tool by its own name and say who runs it.\n\n"
POLICY_TO = "\n\n## The answering agent's tools"


def config(domain: str) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load((BENCH_DIR / domain / "bench.yaml").read_text())
    return cfg


def options(domain: str) -> list[str]:
    root = BENCH_DIR / domain
    return sorted(p.name for p in root.iterdir() if (p / "option.yaml").is_file())


def load_option(domain: str, opt: str) -> rag_agent.RagAgent:
    """An option as a RAG agent: its base rN with the option's merge, the shared reviewer and the
    option's rubric. Its library lives under the name `bench_<opt>`."""
    d = BENCH_DIR / domain / opt
    cfg = yaml.safe_load((d / "option.yaml").read_text())
    base = rag_agent.load(domain, str(cfg["base"]))
    merge = d / "merge.md"
    return dataclasses.replace(
        base,
        name=f"bench_{opt}",
        merge_prompt=merge.read_text() if merge.is_file() else base.merge_prompt,
        review_prompt=(BENCH_DIR / domain / "review.md").read_text(),
        rubric=tuple(str(x) for x in cfg.get("rubric") or base.rubric),
        merge_base="draft",
        review_binding=bool(cfg.get("review_binding", True)),
        merge_steps=int(cfg.get("merge_steps", base.merge_steps)),  # s31 (O4)
        merge_write=str(cfg.get("merge_write", base.merge_write)),  # s31 (O1)
        **rag_agent.merge_settings(cfg, d),
    )


def policy_of(system: str) -> str:
    """The answering agent's policy as a research system prompt prints it."""
    i = system.find(POLICY_FROM)
    return system[i + len(POLICY_FROM) : system.find(POLICY_TO, i)] if i >= 0 else ""


def replay_for(domain: str, job: str, source: str, system: str) -> tuple[rag_agent.Replay, str]:
    """One source session cut to the bench's job: its research as it ran, its draft holding only
    the drafted job its decision names for `job`, and that one decision. Returns it and the
    drafted job's name."""
    src = rag_agent.session(domain, source)
    full = rag_agent._replay(src, system, rag_agent.SCRIPT_OPENING)
    drafted = next(k for k, v in full.decision.items() if job in v or k == job)
    draft = {**full.draft, "jobs": [j for j in full.draft["jobs"] if j.get("job") == drafted]}
    return rag_agent.Replay(full.source, full.messages, draft, {drafted: [job]}), drafted


def fresh_store(domain: str, name: str, seed: str) -> Path:
    """A scratch library seeded exactly as the replayed version's was (its pinned seed.json)."""
    d = store.LIBRARY_DIR / domain / name
    d.mkdir(parents=True)
    shutil.copy(ROOT / "rag_library" / domain / seed / "seed.json", d / "seed.json")
    return d


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def run(
    domain: str,
    job: str,
    opt: str,
    trial: int,
    *,
    ask: rag_agent.Ask | None = None,
    env: Any = None,
    tools: list[dict[str, Any]] | None = None,
    run_tool: rag_agent.RunTool | None = None,
    root: Path | None = None,
    stamp: str | None = None,
    resume: Path | None = None,
    why: str = "",
    upto: int | None = None,
) -> Path:
    """Replay `job`'s merges under option `opt`, one after another, on a scratch library. Returns
    the run folder. A live run (no `ask`) builds the research tools' environment, since a merge
    turn may search the documents again. `resume` continues a run that stopped on a fault in the
    harness's code (never a usage limit), from the merge that failed, on the library it left; the
    failed merge's session stays in the folder and the record says why it was resumed.

    s31 (O8): `upto` stops the run once that many of its merges are done, so its heads are scored
    before the rest is spent; it stops as a pause does, and resumes the same way."""
    from tau2_loop.llm import core

    cfg = config(domain)["jobs"][job]
    agent = load_option(domain, opt)
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = resume or (root or RUNS_DIR) / f"{stamp}_{domain}_{opt}_t{trial}"
    if resume is None:
        out.mkdir(parents=True, exist_ok=False)
    store.LIBRARY_DIR = out / "rag_library"  # every store built from here on is the scratch one
    if resume is None:
        fresh_store(domain, agent.name, str(cfg["seed"]))
    system = (rag_agent.RAG_RUNS_DIR / str(cfg["system_from"]) / "system.md").read_text()
    if ask is None:
        from tau2_loop.llm import require_live

        require_live()
        env = env if env is not None else rag_agent.environment(agent.retrieval)
    if env is not None:
        _, tools = rag_agent.system_prompt(agent, env)
        run_tool = run_tool or rag_agent.env_runner(env)
    policy = str(getattr(env, "policy", "") or "") or policy_of(system)
    record: dict[str, Any] = {
        "id": out.name,
        "domain": domain,
        "job": job,
        "option": opt,
        "trial": trial,
        "agent": agent.name,
        "base": yaml.safe_load((BENCH_DIR / domain / opt / "option.yaml").read_text())["base"],
        "model": core.resolve_model(agent.model),
        "effort": agent.effort,
        "stages": list(agent.stages) if agent.stages else None,
        "reviewer": list(agent.reviewer) if agent.reviewer else None,
        "merge_mode": agent.merge_mode,
        "merge_account": agent.merge_account,
        "merge_write": agent.merge_write,
        "merge_steps": agent.merge_steps,
        "send_back": agent.send_back,
        "held_only": agent.held_only,
        "review_once": agent.review_once,
        "post_audit": agent.post_audit,
        "review_binding": agent.review_binding,
        "rubric": list(agent.rubric),
        "prompts": {
            "merge": _sha(agent.merge_prompt),
            "review": _sha(agent.review_prompt),
            **{s: _sha(t) for s, t in agent.stage_prompts},
        },
        "seed": cfg["seed"],
        "order": cfg["order"],
        "git_sha": _git_sha(),
        "started": stamp,
        "merges": [],
        "status": "running",
    }
    todo = list(cfg["order"])
    if resume is not None:
        record = json.loads((out / "run.json").read_text())
        done = [m["task"] for m in record["merges"] if m["status"] == "done"]
        todo = [t for t in cfg["order"] if t not in done]
        record.setdefault("resumed", []).append(
            {"at": stamp, "git_sha": _git_sha(), "from": todo[0] if todo else None, "why": why}
        )
        record["status"], record["error"] = "running", None

    def save() -> None:
        (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")

    save()
    for task in todo:
        n_done = sum(m["status"] == "done" for m in record["merges"])
        if upto is not None and n_done >= upto:
            record["status"] = "stopped"
            record["error"] = f"stopped by the protocol after {n_done} merges (O8), before {task}"
            save()
            return out
        source = str(cfg["sessions"][task])
        replay, drafted = replay_for(domain, job, source, system)
        sid = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{domain}_{agent.name}_{task}"
        d = out / "sessions" / sid
        d.mkdir(parents=True)
        meta = rag_agent.SessionMeta(
            id=sid,
            domain=domain,
            agent=agent.name,
            task_id=task,
            model=core.resolve_model(agent.model),
            effort=agent.effort,
            retrieval=agent.retrieval,
            started=sid.split("_", 1)[0],
            source="remerge",
            drafts_of=source,
        )
        src_events = rag_agent.session(domain, source)["events"]
        meta = rag_agent.run_session(
            d,
            meta,
            agent,
            system,
            tools or [],
            str(src_events[0].get("question") or ""),
            ask or rag_agent.default_ask,
            run_tool or (lambda name, args: "(no research tools in this run)"),
            policy,
            rag_agent.SCRIPT_OPENING,
            replay,
        )
        got = json.loads((d / "output.json").read_text()) if (d / "output.json").is_file() else {}
        lock = got.get("lock") or {}
        record["merges"].append(
            {
                "task": task,
                "session": sid,
                "source": source,
                "drafted": drafted,
                "status": meta.status,
                "error": meta.error,
                "read": (lock.get("read") or {}).get(job),
                "committed": (lock.get("committed") or {}).get(job),
                "taken": (lock.get("committed") or {}).get(job) is not None,
                "input_tokens": meta.input_tokens,
                "output_tokens": meta.output_tokens,
                "duration_ms": meta.duration_ms,
                "steps": meta.steps,
            }
        )
        save()
        if meta.status != "done":
            record["status"] = "stopped"
            record["error"] = f"{task}: {meta.error}"
            save()
            return out
    record["status"] = "done"
    record["finished"] = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    save()
    return out


def heads(run_dir: Path, job: str) -> list[dict[str, Any]]:
    """Every version of `job` a run's library held, in order: the seed's (v0), then each commit."""
    from tau2_loop.workflows import library

    rec = json.loads((run_dir / "run.json").read_text())
    st = store.FileStore(
        rec["domain"], rec["agent"], load=library._session_job, root=run_dir / "rag_library"
    )
    out = [
        {"version": 0, "task": None, "workflow": e["workflow"]}
        for e in st.entries(upto=0)
        if e["job"] == job
    ]
    for c in st.commits():
        if c["job"] == job:
            v = int(c["version"])
            wf = json.loads((st.dir / "versions" / job / f"{v}.json").read_text())
            out.append({"version": v, "task": c.get("task"), "workflow": wf})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tau2_loop.workflows.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="replay a job's merges under one option (calls the model)")
    r.add_argument("--domain", default="banking_knowledge")
    r.add_argument("--job", default="filing_a_debit_card_transaction_dispute")
    r.add_argument("--option", required=True)
    r.add_argument("--trial", type=int, default=1)
    r.add_argument("--upto", type=int, help="stop once this many merges are done (O8)")
    c = sub.add_parser(
        "resume", help="continue a run that stopped on a code fault (calls the model)"
    )
    c.add_argument("run_dir")
    c.add_argument("--why", required=True)
    c.add_argument("--upto", type=int, help="stop once this many merges are done (O8)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(run(a.domain, a.job, a.option, a.trial, upto=a.upto))
    elif a.cmd == "resume":
        d = Path(a.run_dir)
        rec = json.loads((d / "run.json").read_text())
        if rec["status"] != "stopped" or "limit" in str(rec.get("error") or "").lower():
            raise SystemExit(
                f"{d.name} is {rec['status']}: only a run stopped by a code fault resumes"
            )
        print(
            run(
                rec["domain"],
                rec["job"],
                rec["option"],
                rec["trial"],
                resume=d,
                why=a.why,
                upto=a.upto,
            )
        )


if __name__ == "__main__":
    main()
