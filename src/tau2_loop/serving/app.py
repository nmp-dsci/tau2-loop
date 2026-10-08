"""The API behind the frontend, and the frontend itself when a build is present.

Every read route serves a committed file: `runs/`, `agents/`, `loop/`, `data/`.
No route calls a model, except the live demos (below). The one write route is `POST /api/review/…`, which
records a person's verdict on a conversation the judge already scored — the one
fact here that cannot be a committed file — and it is refused unless the central
Postgres is reachable and this is not the demo image. The demo image is this
process with `DEMO_MODE=1` and nothing else.

`POST /api/runs/…/tool` is a POST that writes nothing: the Agent tab's playground
runs one tool call in a throwaway copy of tau2's environment (`eval/replay.py`).
It needs tau2, which the demo image does not ship, so there it answers 503.

`POST /api/live/<domain>/<agent>` plays one train task live (s16): a version against the
customer, on the subscription, in a background thread that writes `live_runs/<id>/`, scored by
tau2 at the end; never a run, so never logged, gated or graded into a ledger.

`POST /api/workflows/<domain>/rag/sessions` is the other route that calls a model: workflow_rag's
live demo (s16) starts a RAG-agent session on a train question, on the subscription, in a
background thread that writes `rag_agent_runs/<id>/`. The demo image refuses it (no tau2, and
`DEMO_MODE` refuses any model call); a test question is refused everywhere.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tau2_loop import __version__
from tau2_loop.agent.compose import compose, hooks_defined, load_code_surfaces, load_helper_file
from tau2_loop.agent.versions import (
    CODE_SURFACES,
    FROZEN,
    SURFACES,
    AgentVersion,
    list_versions,
    load_version,
)
from tau2_loop.config import (
    BANKING_RETRIEVAL,
    DATASET_AGENTS,
    DOMAINS,
    FRONTEND_DIST,
    RUNS_DIR,
    SMOKE_DOMAIN,
    settings,
)
from tau2_loop.data import documents as kb_docs
from tau2_loop.data import leaderboard as board
from tau2_loop.data import pg
from tau2_loop.data.goals import customer_goal, is_placeholder
from tau2_loop.data.splits import read_split, read_task_extract, split_ids
from tau2_loop.eval import live as live_mod
from tau2_loop.eval import replay
from tau2_loop.eval import retrieval as retrieval_mod
from tau2_loop.eval import review as review_store
from tau2_loop.eval.compare import compare
from tau2_loop.eval.health import run_health
from tau2_loop.eval.profile import profile
from tau2_loop.eval.results import TaskResult, check_counts, read_results, slug
from tau2_loop.eval.runner import RunMeta, list_runs, load_run
from tau2_loop.loop.history import version_history
from tau2_loop.loop.ledger import read_ledger
from tau2_loop.tooljudge import gold as judge_gold_mod
from tau2_loop.tooljudge import labels as judge_labels
from tau2_loop.tooljudge import replay as judge_replay_mod
from tau2_loop.tooljudge import review as judge_review
from tau2_loop.tooljudge import view as judge_view
from tau2_loop.tracking.registry import read_all, read_registry
from tau2_loop.tracking.snapshot import read_snapshot
from tau2_loop.workflows import golden as workflow_golden
from tau2_loop.workflows import rag_agent as workflow_rag


class ToolIn(BaseModel):
    """One playground call: a tool, its arguments, and the message whose state it runs in."""

    name: str
    arguments: dict[str, Any] = {}
    at: int
    after_calls: int = 0
    requestor: str = "assistant"


class LiveIn(BaseModel):
    """A live conversation: a train task, and the customer's model unless the runs' default."""

    task_id: str
    user_model: str = ""


class RagSessionIn(BaseModel):
    """A workflow_rag demo session: a train question, and the version's model and effort unless set."""

    task_id: str
    agent: str = "r1"
    model: str | None = None
    effort: str | None = None


class GoldReviewIn(BaseModel):
    """A person's check of one golden answer (s11 J1): agree, or correct with what it should be."""

    item_id: str
    verdict: str
    correction: dict[str, Any] = {}
    note: str = ""
    author: str = ""


class ReviewIn(BaseModel):
    """One recorded verdict. `verdict` is checked again by the table's own CHECK."""

    verdict: str
    reason: str = ""
    author: str = ""


def create_app() -> FastAPI:
    app = FastAPI(title="tau2-loop", version=__version__)
    s = settings()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        champs = {d: (read_registry(d).get("champion") or {}).get("agent") for d in DOMAINS}
        return {
            "status": "ok",
            "mode": "demo" if s.demo_mode else "live",
            "champions": champs,
            "code_sha": s.code_sha,
            "version": __version__,
            "domains": list(DOMAINS),
            "mlflow_url": settings().mlflow_tracking_uri,
            "mlflow_embeddable": _mlflow_embeddable(),
            # whether a person can record a review here: the demo image cannot, and
            # neither can a checkout with the central Postgres stopped
            "writable": (not s.demo_mode) and pg.reachable(),
            # whether the Agent tab's playground can run: it needs tau2, which the image lacks
            "playground": replay.available(),
        }

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        """One object the overview is built from: how big the benchmark is, how much
        of it we have run, and what that cost. Every number comes from a committed file."""
        runs = list_runs()
        real = [m for m in runs if not m.dry_run and m.domain in DOMAINS]
        scored = [m for m in real if (m.summary or {}).get("n_scored")]
        per_domain = []
        for d in DOMAINS:
            try:
                split = read_split(d)
            except FileNotFoundError:
                split = {}
            reg = read_registry(d)
            champ = reg.get("champion") or {}
            per_domain.append(
                {
                    "domain": d,
                    "base_n": split.get("base_n"),
                    "train": len(split.get("train", [])),
                    "test": len(split.get("test", [])),
                    "reserve_n": split.get("reserve_n") or 0,
                    "champion": champ.get("agent"),
                    "champion_run": champ.get("run_id"),
                    "passed": champ.get("passed"),
                    "n_scored": champ.get("n_scored"),
                    "runs": sum(1 for m in real if m.domain == d),
                    "cycles": len(read_ledger(d)),
                    "versions": len(list_versions(d)),
                }
            )
        conversations = sum(int((m.summary or {}).get("n") or 0) for m in scored)
        return {
            "domains": per_domain,
            "base_total": sum(x["base_n"] or 0 for x in per_domain),
            "runs": len(real),
            "runs_scored": len(scored),
            "conversations": conversations,
            "cost_usd_est": round(
                sum(float((m.summary or {}).get("cost_usd_est") or 0.0) for m in scored), 2
            ),
            "cycles": sum(x["cycles"] for x in per_domain),
            "code_sha": s.code_sha,
            "mode": "demo" if s.demo_mode else "live",
        }

    @app.get("/api/rubric")
    def rubric() -> dict[str, Any]:
        """How τ² decides a conversation passed, and how often each mechanism is the
        one that failed. The first half is the task set; the second is our own runs."""
        kinds = []
        for d in DOMAINS:
            ext = read_task_extract(d)
            tasks = ext.get("tasks", [])
            counts = dict.fromkeys(
                ("db_check", "actions", "communicate_info", "nl_assertions", "env_assertions"), 0
            )
            for t in tasks:
                ev = t.get("evaluation_criteria") or {}
                for k in ("actions", "communicate_info", "nl_assertions", "env_assertions"):
                    if ev.get(k):
                        counts[k] += 1
                # the database is not a criterion the task lists: it is named by the basis
                if "DB" in (ev.get("reward_basis") or []):
                    counts["db_check"] += 1
            kinds.append(
                {
                    "domain": d,
                    "n_tasks": len(tasks),
                    "uses": counts,
                    "reward_bases": _basis_counts(ext),
                }
            )
        return {"by_domain": kinds, "failures": _failure_mix(list_runs())}

    # ── benchmark data ────────────────────────────────────────────────────
    @app.get("/api/domains")
    def domains() -> list[dict[str, Any]]:
        out = []
        for d in DOMAINS:
            ext = read_task_extract(d)
            try:
                split = read_split(d)
            except FileNotFoundError:
                split = {}
            reg = read_registry(d)
            out.append(
                {
                    "domain": d,
                    "base_n": split.get("base_n"),
                    "train": len(split.get("train", [])),
                    "test": len(split.get("test", [])),
                    "reserve_n": split.get("reserve_n"),
                    "seed": split.get("seed"),
                    "split_version": split.get("version", 1),
                    "policy_words": ext.get("policy_words"),
                    "n_tools": len(ext.get("tools", [])),
                    "reward_bases": _basis_counts(ext),
                    "champion": reg.get("champion"),
                    "challenger": reg.get("challenger"),
                    "versions": [v.name for v in list_versions(d)],
                    "runs": len(list_runs(d)),
                    "cycles": len(read_ledger(d)),
                }
            )
        return out

    @app.get("/api/domains/{domain}")
    def domain(domain: str) -> dict[str, Any]:
        _check_domain(domain)
        ext = read_task_extract(domain)
        split = read_split(domain)
        return {
            "domain": domain,
            "split": split,
            "policy": ext.get("policy"),
            "policy_words": ext.get("policy_words"),
            "tools": ext.get("tools"),
            # the tools as the domain's champion is given them: banking's retrieval is a version
            # setting, and the extract lists the default variant's (s13)
            "harness_tools": _harness_tools(domain, ext.get("tools") or []),
            "tasks": [_task_row(t, split) for t in ext.get("tasks", [])],
        }

    @app.get("/api/domains/{domain}/conversations")
    def task_conversations(domain: str, task: str) -> list[dict[str, Any]]:
        """Every conversation of one task across the domain's runs, newest run first: the task's
        trace in each experiment, which the scope bar's task opens on Runs."""
        _check_domain(domain)
        out: list[dict[str, Any]] = []
        for m in reversed(list_runs(domain)):
            if m.dry_run or task not in (m.task_ids or [task]):
                continue
            try:
                _, results = load_run(m.run_id)
            except FileNotFoundError:
                continue
            out += [
                {
                    "run_id": m.run_id,
                    "agent": m.agent,
                    "split": m.split,
                    "model": m.model,
                    "started_at": m.started_at,
                    "trial": r.trial,
                    "correct": r.correct,
                    "reward": r.reward,
                    "db_check": r.db_check,
                    "action_checks": r.action_checks,
                    "communicate_checks": r.communicate_checks,
                    "n_agent_turns": r.n_agent_turns,
                    "n_tool_calls": r.n_tool_calls,
                    "termination_reason": r.termination_reason,
                    "duration_ms": r.duration_ms,
                }
                for r in results
                if r.task_id == task
            ]
        return out

    @app.get("/api/domains/{domain}/documents/{doc_id}")
    def kb_document(domain: str, doc_id: str) -> dict[str, Any]:
        """One knowledge-base document's full text, read from tau2's files; the demo image ships
        without tau2, so there only the committed index (title, size) is known."""
        _check_domain(domain)
        known = kb_docs.read_index(domain)
        if doc_id not in known:
            raise HTTPException(404, "no such document")
        doc = kb_docs.document(domain, doc_id)
        if doc is None:
            raise HTTPException(404, "the document's text is not in this image (it ships no tau2)")
        return doc

    @app.get("/api/domains/{domain}/tasks/{task_id:path}/reads")
    def task_reads(domain: str, task_id: str) -> dict[str, Any]:
        """Which of a task's required documents each version reached, in its newest scored run
        that played the task: `read` (shown whole), `seen` (named in a result only), from the
        run's harness health (`eval/health.py`). A version with no such run is left out."""
        _check_domain(domain)
        spec = _task_spec(domain, task_id)
        required = list(spec.get("required_documents") or [])
        out: list[dict[str, Any]] = []
        metas = list_runs(domain)
        for v in list_versions(domain):
            m = next(
                (
                    m
                    for m in reversed(metas)
                    if m.agent == v.name
                    and not m.dry_run
                    and (m.summary or {}).get("n_scored")
                    and task_id in (m.task_ids or [])
                    and (RUNS_DIR / m.run_id / "tau2_results.json").is_file()
                ),
                None,
            )
            if m is None:
                continue
            try:
                _, results = load_run(m.run_id)
            except FileNotFoundError:
                continue
            row = next((r for r in results if r.task_id == task_id), None)
            if row is None:
                continue
            h = _conversation_health(m.run_id, task_id, row.trial) or {}
            out.append(
                {
                    "version": v.name,
                    "run_id": m.run_id,
                    "split": m.split,
                    "retrieval": m.retrieval,
                    "trial": row.trial,
                    "trials": m.trials,
                    "passed": bool(row.correct),
                    "read": list(h.get("required_docs_read_ids") or []),
                    "seen": list(h.get("required_docs_seen_ids") or []),
                    "health": bool(h),
                }
            )
        return {"domain": domain, "task_id": task_id, "required": required, "versions": out}

    @app.get("/api/domains/{domain}/tasks/{task_id:path}")
    def task(domain: str, task_id: str) -> dict[str, Any]:
        _check_domain(domain)
        t = _task_spec(domain, task_id)
        return {
            **t,
            "split": _which_split(task_id, read_split(domain)),
            # banking: each required document's title and size, from the committed index
            "documents": kb_docs.described(domain, list(t.get("required_documents") or [])),
        }

    # ── agents and runs ──────────────────────────────────────────────────
    @app.get("/api/agents")
    def agents(domain: str | None = None) -> dict[str, Any]:
        """Every version with its architecture read from its own folder, so the Agent tab can
        draw a version that has no run, beside its parent (s13 §2)."""
        out = []
        for d in DOMAINS if domain is None else (domain,):
            versions = list_versions(d)
            names = [v.name for v in versions]
            cycles = _cycles(d)
            metas = list_runs(d)
            for v in versions:
                out.append(
                    {
                        "domain": d,
                        "name": v.name,
                        "ref": v.ref,
                        "fingerprint": v.fingerprint,
                        "config": v.config.__dict__,
                        "has_helper": v.helper is not None,
                        "helper_functions": re.findall(r"^def (\w+)", v.helper or "", re.M),
                        "runs": [m.run_id for m in metas if m.agent == v.name],
                        **_architecture(v, names, cycles),
                    }
                )
        # every agent lives under a dataset (s16): the viewer lists only these
        kinds = {d: list(DATASET_AGENTS.get(d, ("answering",))) for d in DOMAINS}
        return {"versions": out, "registry": read_all(), "kinds": kinds}

    @app.get("/api/workflows/{domain}/golden")
    def workflow_golden_set(domain: str) -> dict[str, Any]:
        """workflow_rag's golden set (s16): train entries open, test as sealed counts."""
        if domain not in DOMAINS:
            raise HTTPException(404, "no such domain")
        g = workflow_golden.golden_set(domain)
        if g is None:
            raise HTTPException(404, f"{domain} has no workflow_rag golden set")
        return g

    # before `/api/live/{domain}/{agent}`, which would otherwise take "conversations" as a domain
    @app.get("/api/live/conversations/{live_id}")
    def live_get(live_id: str, since: int = 0) -> dict[str, Any]:
        """A live conversation's events from `since` on, and where the next poll starts."""
        try:
            return live_mod.conversation(live_id, max(0, since))
        except live_mod.LiveError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/live/{domain}/{agent}")
    def live_list(domain: str, agent: str) -> dict[str, Any]:
        """A version's live conversations, and the train tasks one can be started on."""
        _check_domain(domain)
        try:
            train = set(split_ids(domain, "train"))
        except FileNotFoundError:
            train = set()
        tasks = [
            {
                "id": t["id"],
                "goal": customer_goal(((t.get("user_scenario") or {}).get("instructions")) or ""),
            }
            for t in read_task_extract(domain)["tasks"]
            if t["id"] in train
        ]
        ok = replay.available() and not s.demo_mode
        return {
            "conversations": live_mod.conversations(domain, agent),
            "tasks": tasks,
            "live": ok,
            "reason": ""
            if ok
            else "the demo image calls no model and ships no tau2: run the viewer from a checkout (`make dev`)",
        }

    @app.post("/api/live/{domain}/{agent}")
    def live_start(domain: str, agent: str, body: LiveIn) -> dict[str, Any]:
        """Play one train task live; it runs in the background and the viewer polls it."""
        _check_domain(domain)
        if s.demo_mode or not replay.available():
            raise HTTPException(503, "the demo image calls no model and ships no tau2")
        try:
            meta = live_mod.start(domain, agent, body.task_id, body.user_model)
        except live_mod.LiveError as e:
            raise HTTPException(422, str(e)) from e
        except RuntimeError as e:  # billing refused
            raise HTTPException(503, str(e)) from e
        return {"id": meta.id}

    def _rag_domain(domain: str) -> None:
        if domain not in DOMAINS or not workflow_golden.has_workflow_rag(domain):
            raise HTTPException(404, f"{domain} has no workflow_rag agent")

    @app.get("/api/workflows/{domain}/rag")
    def workflow_rag_agents(domain: str) -> dict[str, Any]:
        """workflow_rag's versions with their settings, prompt and tools, and its sessions."""
        _rag_domain(domain)
        names = workflow_rag.versions(domain)
        return {
            "versions": [workflow_rag.describe(domain, n) for n in names],
            "sessions": workflow_rag.sessions(domain),
            "live": replay.available() and not s.demo_mode,
            "reason": ""
            if replay.available() and not s.demo_mode
            else "the demo image calls no model and ships no tau2: run the viewer from a checkout (`make dev`)",
        }

    @app.post("/api/workflows/{domain}/rag/sessions")
    def workflow_rag_start(domain: str, body: RagSessionIn) -> dict[str, Any]:
        """Start a RAG-agent session on a train question; it runs in the background and the
        viewer polls it. The one route that calls a model (on the subscription)."""
        _rag_domain(domain)
        if s.demo_mode or not replay.available():
            raise HTTPException(503, "the demo image calls no model and ships no tau2")
        try:
            meta = workflow_rag.start(domain, body.task_id, body.agent, body.model, body.effort)
        except workflow_rag.RagAgentError as e:
            raise HTTPException(422, str(e)) from e
        except RuntimeError as e:  # billing refused: a key that would bill per token
            raise HTTPException(503, str(e)) from e
        return {"id": meta.id}

    @app.get("/api/workflows/{domain}/rag/library")
    def workflow_rag_library(domain: str, version: str) -> dict[str, Any]:
        """A RAG version's library now (s21): each job, its version and the question that wrote it,
        and, for a concurrent version, which workers hold or wait for which jobs."""
        _rag_domain(domain)
        from tau2_loop.workflows import library as wf_library

        try:
            agent = workflow_rag.load(domain, version)
        except workflow_rag.RagAgentError as e:
            raise HTTPException(404, str(e)) from e
        concurrent = agent.merge == "concurrent"
        st = wf_library.store_for(domain, version) if concurrent else None
        jobs = [
            {
                "job": e["job"],
                "version": e.get("version"),
                "session": e["session"],
                "task": e.get("task_id"),
            }
            for e in wf_library.entries(domain, version)
        ]
        locks = st.locks() if st else {"held": [], "waiting": []}
        return {
            "version": version,
            "concurrent": concurrent,
            "workers": agent.workers,
            "commits": len(st.commits()) if st else None,
            "jobs": jobs,
            **locks,
        }

    @app.get("/api/workflows/{domain}/rag/sessions/{session_id}")
    def workflow_rag_session(domain: str, session_id: str) -> dict[str, Any]:
        """One session as it stands: its events so far, and its workflow and score once done."""
        _rag_domain(domain)
        try:
            return workflow_rag.session(domain, session_id)
        except workflow_rag.RagAgentError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/agents/diff")
    def agents_diff(domain: str, a: str, b: str) -> dict[str, Any]:
        """Unified diff of the two surfaces between versions, with the reasoning that produced `b`."""
        try:
            va, vb = load_version(domain, a), load_version(domain, b)
        except FileNotFoundError as e:
            raise HTTPException(404, "no such agent") from e
        fa, fb = va.files(), vb.files()
        files = []
        # a code surface (s09) is listed only where one of the two versions has it
        for name in (*SURFACES, *FROZEN):
            if name in CODE_SURFACES and name not in fa and name not in fb:
                continue
            ta, tb = fa.get(name, ""), fb.get(name, "")
            lines = list(
                difflib.unified_diff(
                    ta.splitlines(),
                    tb.splitlines(),
                    fromfile=f"{a}/{name}",
                    tofile=f"{b}/{name}",
                    lineterm="",
                    n=3,
                )
            )
            files.append(
                {
                    "name": name,
                    "changed": ta != tb,
                    "added": sum(1 for ln in lines[2:] if ln.startswith("+")),
                    "removed": sum(1 for ln in lines[2:] if ln.startswith("-")),
                    "diff": lines,
                    "before": ta,
                    "after": tb,
                }
            )
        diag = vb.path / "diagnosis.json"
        diagnosis = json.loads(diag.read_text()) if diag.exists() else None
        transcript_path = vb.path / "optimiser_transcript.json"
        closing = None
        if transcript_path.exists():
            prose = [
                m["content"]
                for m in json.loads(transcript_path.read_text())
                if m.get("role") == "assistant"
            ]
            closing = prose[-1] if prose else None
        cycles = [e for e in read_ledger(domain) if e.get("challenger") == b]
        return {
            "domain": domain,
            "a": {
                "name": a,
                "fingerprint": va.fingerprint,
                "runs": [m.__dict__ for m in list_runs(domain) if m.agent == a],
            },
            "b": {
                "name": b,
                "fingerprint": vb.fingerprint,
                "runs": [m.__dict__ for m in list_runs(domain) if m.agent == b],
            },
            "files": files,
            "diagnosis": diagnosis,
            "closing_account": closing,
            "cycles": cycles,
        }

    @app.get("/api/agents/{domain}/{name}")
    def agent(domain: str, name: str) -> dict[str, Any]:
        try:
            v = load_version(domain, name)
        except FileNotFoundError as e:
            raise HTTPException(404, "no such agent") from e
        return {
            "domain": domain,
            "name": v.name,
            "fingerprint": v.fingerprint,
            "config": v.config.__dict__,
            "files": v.files(),
            **_architecture(v, [x.name for x in list_versions(domain)], _cycles(domain)),
        }

    @app.get("/api/runs")
    def runs(domain: str | None = None) -> list[dict[str, Any]]:
        out = []
        for m in list_runs(domain):
            path = RUNS_DIR / m.run_id / "results.jsonl"
            results = read_results(path) if path.exists() else []
            out.append(
                {
                    **m.__dict__,
                    "mlflow_url": _mlflow_run_url(m.mlflow_run_id),
                    "tokens_per_conversation": _tokens_per_conversation(results),
                    # a run scored before Summary.checks existed is aggregated here, from its rows
                    "checks": (m.summary or {}).get("checks") or check_counts(results),
                    "split_version": m.split_version or _cut_of(m),
                }
            )
        return out

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        try:
            meta, results = load_run(run_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "no such run") from e
        h = _health(run_id)
        return {
            "meta": {**meta.__dict__, "mlflow_url": _mlflow_run_url(meta.mlflow_run_id)},
            "results": [r.__dict__ for r in results],
            "profile": profile(results),
            # s13: how the agent used its harness (eval/health.py); None without tau2's results
            "health": h["summary"] if h else None,
        }

    @app.get("/api/runs/{run_id}/traces/{name}")
    def trace(run_id: str, name: str) -> dict[str, Any]:
        p = RUNS_DIR / run_id / "traces" / name
        if "/" in name or not p.is_file():
            raise HTTPException(404, "no trace")
        t = json.loads(p.read_text())
        return {
            "task_id": t.get("task_id"),
            "trial": t.get("trial"),
            "termination_reason": t.get("termination_reason"),
            "duration": t.get("duration"),
            "agent_cost": t.get("agent_cost"),
            "user_cost": t.get("user_cost"),
            "reward_info": t.get("reward_info"),
            "events": trace_events(t),
            "policy_words": len(str(t.get("policy") or "").split()),
        }

    @app.get("/api/runs/{run_id}/agent")
    def run_agent(run_id: str) -> dict[str, Any]:
        """The agent a run ran with, from the run's own snapshot (`runs/<id>/agent/`), not
        `agents/`: the run folder is the record. The prompt is composed by the same
        function the agent uses, with the policy the trace recorded."""
        try:
            meta, results = load_run(run_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "no such run") from e
        snap = RUNS_DIR / run_id / "agent"
        files = {n: (snap / n).read_text() for n in (*SURFACES, *FROZEN) if (snap / n).is_file()}
        if "system.md" not in files:
            raise HTTPException(404, "this run has no agent snapshot")
        policy = _recorded_policy(run_id, [r.trace for r in results]) or str(
            read_task_extract(meta.domain).get("policy") or ""
        )
        helper = load_helper_file(
            snap / "helper.py" if "helper.py" in files else None, f"tau2_loop_run_helper_{run_id}"
        )
        config = yaml.safe_load(files.get("agent.yaml") or "") or {}
        # a run before the clock note (no `sim_rules`) was sent no note, and a version without
        # `identity_note: true` no identity note: show what it was sent
        c = compose(
            files["system.md"],
            policy,
            helper,
            clock=meta.sim_rules is not None,
            identity=config.get("identity_note") is True,
        )
        return {
            "run_id": run_id,
            "domain": meta.domain,
            "agent": meta.agent,
            "fingerprint": meta.fingerprint,
            "config": config,
            "files": files,
            "hooks": hooks_defined(
                helper, load_code_surfaces(snap, f"tau2_loop_run_code_{run_id}")
            ),
            "prompt": {
                "text": c.text,
                "system_md_chars": len(c.system_md),
                "policy": c.policy,
                "policy_words": len(c.policy.split()),
                "extra_context": c.extra_context,
                "clock_note": c.clock_note,
                "identity_note": c.identity_note,
                "slotted": c.slotted,
            },
        }

    @app.get("/api/runs/{run_id}/{task_id}/{trial}")
    def trial(run_id: str, task_id: str, trial: str) -> dict[str, Any]:
        """One conversation, addressed the way the viewer addresses it: task id and
        `t<n>`. The trace file name (`eval/runner.py` slugs the task id) stays on disk.

        Beside the flat `events` the Trace page reads, it carries every message whole
        (ids, so a result pairs with its call; usage; timing), the task spec, and the
        domain's tools with tau2's own read/write type, for the Agent tab's graph."""
        meta, row = _conversation(run_id, task_id, trial)
        t = json.loads((RUNS_DIR / run_id / "traces" / row.trace).read_text())
        ext = read_task_extract(meta.domain)
        spec = next((x for x in ext.get("tasks", []) if x.get("id") == row.task_id), None)
        return {
            **trace(run_id, row.trace),
            "result": row.__dict__,
            "domain": meta.domain,
            "messages": trace_messages(t),
            "task": spec,
            "tools": ext.get("tools") or [],
            "user_tools": ext.get("user_tools") or [],
            # s13: this conversation's harness-health row (eval/health.py), when the run kept
            # tau2's results; tau2 counts trials from 0, the run's rows from 1
            "health": _conversation_health(run_id, row.task_id, row.trial),
            # s11: the tool judge's labels, golden answer and verdicts, on a labelled train conversation
            "judge": _judge_with_checks(meta.domain, run_id, row.task_id, row.trial),
        }

    def _judge_with_checks(
        domain: str, run_id: str, task_id: str, trial: int
    ) -> dict[str, Any] | None:
        """The judge's view of a conversation plus a person's checks not yet frozen into the gold
        file, so a correction made on the conversation shows there at once."""
        j = judge_view.conversation(domain, run_id, task_id, trial)
        if j is None or j.get("gold") is None:
            return j
        db = pg.reachable()
        j["checks"] = judge_review.current_conversation(domain, j["labels"]["key"]) if db else {}
        j["writable"] = db and not s.demo_mode
        j["reason"] = (
            "" if j["writable"] else ("the demo image is read only" if s.demo_mode else _no_db())
        )
        return j

    @app.post("/api/runs/{run_id}/{task_id}/{trial}/tool")
    def playground(run_id: str, task_id: str, trial: str, body: ToolIn) -> dict[str, Any]:
        """Run one tool call against the database as it stood at message `at`.

        Writes nothing: the environment is rebuilt in memory for this call and dropped.
        Calls no model. Answers 503 where tau2 is not installed (the demo image)."""
        if not replay.available():
            raise HTTPException(
                503,
                "the playground needs tau2, which the demo image does not ship: "
                "run the viewer from a checkout (`make dev`)",
            )
        meta, row = _conversation(run_id, task_id, trial)
        try:
            conv = _loaded_conversation(run_id, row.trace, meta.domain, row.task_id)
            return replay.run_tool(
                conv,
                name=body.name,
                arguments=body.arguments,
                at=body.at,
                after_calls=body.after_calls,
                requestor=body.requestor,
            )
        except replay.ReplayError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/runs/{run_id}/{task_id}/{trial}/db")
    def db_check(run_id: str, task_id: str, trial: str) -> dict[str, Any]:
        """Where the conversation's final database differs from gold's: both sides of tau2's DB
        check rebuilt in memory, each differing field with its value before the conversation,
        and the agent's calls and the expected actions that changed each record.

        Writes nothing, calls no model. Answers 503 where tau2 is not installed (the demo image)."""
        if not replay.available():
            raise HTTPException(
                503,
                "the database diff needs tau2, which the demo image does not ship: "
                "run the viewer from a checkout (`make dev`)",
            )
        meta, row = _conversation(run_id, task_id, trial)
        try:
            out = _db_diff(run_id, row.trace, meta.domain, row.task_id)
        except replay.ReplayError as e:
            raise HTTPException(422, str(e)) from e
        # what the run was graded: a rebuild that disagrees with it says so rather than hides it
        return {**out, "graded": row.db_check}

    @app.get("/api/leaderboard")
    def leaderboard() -> dict[str, Any]:
        """The published board, and our own runs on the same axis.

        Every published entry is self-reported: a team runs the harness and opens a
        pull request. Ours are not on it — `ours` is what a submission would claim,
        computed from the same `pass^k` the harness writes into each run's summary.
        """
        index = board.read()
        entries = index.get("entries") or []
        ours = []
        for m in list_runs():
            s = m.summary or {}
            if m.dry_run or not s.get("n_scored") or m.domain not in DOMAINS:
                continue
            ours.append(
                {
                    "run_id": m.run_id,
                    "domain": m.domain,
                    "agent": m.agent,
                    "fingerprint": m.fingerprint,
                    "split": m.split,
                    "n_tasks": m.n_tasks,
                    "trials": m.trials,
                    "model": m.model,
                    "user_model": m.user_model,
                    "pass_hat_k": s.get("pass_hat_k") or {},
                    "passed": s.get("passed"),
                    "n_scored": s.get("n_scored"),
                    "cost_usd_est": s.get("cost_usd_est"),
                    "started_at": m.started_at,
                }
            )
        return {
            **index,
            "best": board.best_per_domain(entries),
            "ours": ours,
            # what would make our entry unverified if it were submitted (plan s03)
            "our_caveats": {
                "user_simulator": "claude-sdk/claude-haiku-4-5",
                "tool_calling": "a JSON contract in the prompt, not native tool calls",
                "prompts": "written by our optimiser, so `modified_prompts` would be true",
                "split": "our own 20/20 train/test cut, not the full base set",
            },
        }

    # ── reviews: the one write path (s04 M5) ─────────────────────────────
    @app.get("/api/review")
    def reviews(run_id: str | None = None) -> dict[str, Any]:
        """Every current verdict, and the tally. Empty — never an error — with no database."""
        if not pg.reachable():
            return {"writable": False, "reason": _no_db(), "current": {}, "tally": {}}
        return {
            "writable": not s.demo_mode,
            "reason": "the demo image is read only" if s.demo_mode else "",
            "current": review_store.current(run_id),
            "tally": review_store.tally(run_id),
        }

    @app.get("/api/review/{run_id}/{task_id}/{trial}")
    def review_one(run_id: str, task_id: str, trial: str) -> dict[str, Any]:
        n = _trial_number(trial)
        if not pg.reachable():
            return {"writable": False, "reason": _no_db(), "history": []}
        return {
            "writable": not s.demo_mode,
            "reason": "the demo image is read only" if s.demo_mode else "",
            "history": review_store.history(run_id, task_id, n),
        }

    @app.post("/api/review/{run_id}/{task_id}/{trial}")
    def review_write(run_id: str, task_id: str, trial: str, body: ReviewIn) -> dict[str, Any]:
        """Record one verdict. Refused in the demo image, and with no database."""
        if s.demo_mode:
            raise HTTPException(403, "the demo image is read only")
        if not pg.reachable():
            raise HTTPException(503, _no_db())
        n = _trial_number(trial)
        try:
            meta, results = load_run(run_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "no such run") from e
        row = next((r for r in results if r.task_id == task_id and r.trial == n), None)
        if row is None:
            raise HTTPException(404, "no such conversation")
        try:
            return review_store.add(
                run_id=run_id,
                task_id=task_id,
                trial=n,
                judge_reward=row.reward,
                verdict=body.verdict,
                reason=body.reason[:4000],
                author=body.author[:120],
                code_sha=meta.code_sha,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/compare")
    def compare_runs(a: str, b: str) -> dict[str, Any]:
        ma, ra = load_run(a)
        mb, rb = load_run(b)
        # The gate refuses runs over different tasks; a page comparing any two runs (an old cut
        # against a new one) shows the overlap instead, and says that is what it shows.
        note = None
        try:
            v: Any = compare(ra, rb).__dict__
        except ValueError as e:
            common = {r.task_id for r in ra} & {r.task_id for r in rb}
            try:
                v = compare(
                    [r for r in ra if r.task_id in common], [r for r in rb if r.task_id in common]
                ).__dict__
                note = f"{e}; the verdict is on the {len(common)} tasks both runs scored"
            except ValueError as e2:
                v, note = None, str(e2)
        rows = []
        da = {(r.task_id, r.trial): r for r in ra}
        db = {(r.task_id, r.trial): r for r in rb}
        for key in sorted(
            set(da) | set(db),
            key=lambda k: (ma.task_ids.index(k[0]) if k[0] in ma.task_ids else 1 << 30, k[1]),
        ):
            rows.append(
                {
                    "task_id": key[0],
                    "trial": key[1],
                    "purpose": (da.get(key) or db.get(key)).purpose,  # type: ignore[union-attr]
                    "a": da[key].__dict__ if key in da else None,
                    "b": db[key].__dict__ if key in db else None,
                }
            )
        return {
            "verdict": v,
            "note": note,
            "rows": rows,
            "a": {**ma.__dict__, "mlflow_url": _mlflow_run_url(ma.mlflow_run_id)},
            "b": {**mb.__dict__, "mlflow_url": _mlflow_run_url(mb.mlflow_run_id)},
            "profiles": {"a": profile(ra), "b": profile(rb)},
        }

    @app.get("/api/ledger")
    def ledger(domain: str | None = None) -> dict[str, list[dict[str, Any]]]:
        return {d: read_ledger(d) for d in (DOMAINS if domain is None else (domain,))}

    @app.get("/api/registry")
    def registry() -> dict[str, Any]:
        return read_all()

    @app.get("/api/versions")
    def versions(domain: str | None = None) -> dict[str, Any]:
        """Per domain, every version in build order with the run the gate read, its test run,
        how it was made and who held the title: the figures on the Runs and Optimise tabs."""
        out = {}
        for d in DOMAINS if domain is None else (domain,):
            runs = [
                {
                    "run_id": m.run_id,
                    "agent": m.agent,
                    "split": m.split,
                    "cut": m.split_version or _cut_of(m),
                    "passed": m.summary["passed"],
                    "n": m.summary["n_scored"],
                    "model": m.model,
                    "effort": m.agent_effort,
                }
                for m in list_runs(d)
                if not m.dry_run and m.finished_at and m.summary and m.summary.get("n_scored")
            ]
            vs = [
                {
                    "name": v.name,
                    "model": v.config.model,
                    "effort": v.config.effort,
                    "diagnosis": _json_or_none(v.path / "diagnosis.json"),
                }
                for v in list_versions(d)
            ]
            out[d] = version_history(d, vs, runs, read_ledger(d), read_registry(d))
        return out

    @app.get("/api/experiments")
    def experiments() -> dict[str, Any]:
        return read_snapshot()

    # ── the tool judge (s11): labels, golden answers, replays — committed files ──
    @app.get("/api/judge/{domain}")
    def judge_overview(domain: str) -> dict[str, Any]:
        """Optimise's judge view: J0's labels, J1's golden answers, every J2 replay scored on them."""
        _check_domain(domain)
        return judge_view.overview(domain)

    @app.get("/api/judge/{domain}/evals")
    def judge_evals(domain: str) -> dict[str, Any]:
        """The LLM judge's eval set: labelled train conversations, golden answers, folds, synthetics."""
        _check_domain(domain)
        return judge_view.evals(domain) or {"domain": domain, "conversations": None}

    @app.get("/api/judge/{domain}/versions")
    def judge_versions(domain: str) -> dict[str, Any]:
        """The judge agent itself: each version's rubric and config, and the SDK probe."""
        _check_domain(domain)
        return {
            "versions": judge_view.versions(domain),
            "probe": judge_labels.read_json(judge_labels.JUDGE_DATA_DIR / "probe.json"),
        }

    @app.get("/api/judge/{domain}/replays")
    def judge_replays(domain: str) -> list[dict[str, Any]]:
        """Every replay folder's run.json, newest first, unscored: cheap enough for a filter."""
        _check_domain(domain)
        return judge_replay_mod.list_replays(domain)

    @app.get("/api/judge/{domain}/gold")
    def judge_gold(domain: str) -> dict[str, Any]:
        """The golden answers' summary and Review's queue, with each item's current check."""
        _check_domain(domain)
        items = judge_review.queue(domain)
        db = pg.reachable()
        return {
            # computed from the committed gold file, so a run still writing it shows its progress
            "summary": judge_gold_mod.summarise(domain)
            if judge_gold_mod.read_gold(domain)
            else None,
            "labels": (judge_labels.read_labels(domain) or {}).get("summary"),
            "items": items,
            "current": judge_review.current(domain) if db else {},
            "writable": db and not s.demo_mode,
            "reason": ""
            if db and not s.demo_mode
            else ("the demo image is read only" if s.demo_mode else _no_db()),
        }

    @app.get("/api/judge/{domain}/gold/item")
    def judge_gold_item(domain: str, id: str) -> dict[str, Any]:  # noqa: A002 - the query's own name
        """One review item: the annotator's answer, the messages it cites, and every check of it."""
        _check_domain(domain)
        item = judge_review.item(domain, id)
        if item is None:
            raise HTTPException(404, "no golden answer at that message")
        return {
            **item,
            "messages": judge_review.item_messages(domain, item),
            "history": judge_review.history(domain, id) if pg.reachable() else [],
        }

    @app.post("/api/review/golden/{domain}")
    def judge_gold_review(domain: str, body: GoldReviewIn) -> dict[str, Any]:
        """Record a person's check of one golden answer. Refused in the demo image and with no database."""
        _check_domain(domain)
        if s.demo_mode:
            raise HTTPException(403, "the demo image is read only")
        if not pg.reachable():
            raise HTTPException(503, _no_db())
        item = judge_review.item(domain, body.item_id)
        if item is None:
            raise HTTPException(404, "no golden answer at that message")
        try:
            return judge_review.add(
                domain,
                body.item_id,
                body.verdict,
                correction=body.correction,
                note=body.note[:4000],
                author=body.author[:120],
                annotator_sha=item["annotator_sha"],
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    _mount_frontend(app)
    return app


def _conversation(run_id: str, task_id: str, trial: str) -> tuple[Any, Any]:
    """The run and the results row for `task_id` / `t<n>`, or a 404 that says which is missing."""
    m = re.fullmatch(r"t(\d+)", trial)
    if not m:
        raise HTTPException(404, "no such trial")
    n = int(m.group(1))
    try:
        meta, results = load_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(404, "no such run") from e
    row = next((r for r in results if r.task_id == task_id and r.trial == n), None)
    if row is None:
        # the pre-grammar address carried a slugged task id; fall back to the slug
        row = next((r for r in results if slug(r.task_id) == task_id and r.trial == n), None)
    if row is None or not row.trace:
        raise HTTPException(404, "no such conversation")
    return meta, row


@lru_cache(maxsize=64)
def _db_diff(run_id: str, trace_name: str, domain: str, task_id: str) -> dict[str, Any]:
    """A conversation's database diff against gold. Run folders never change once scored, so
    it is computed once per conversation (about a second: a replay per write)."""
    return replay.db_diff(_loaded_conversation(run_id, trace_name, domain, task_id))


@lru_cache(maxsize=32)
def _loaded_conversation(run_id: str, trace_name: str, domain: str, task_id: str) -> Any:
    """A committed conversation as tau2 objects. Run folders never change once scored,
    so a parsed trace is safe to keep; the environment built from it never is."""
    return replay.load_conversation(run_id, trace_name, domain, task_id)


def _recorded_policy(run_id: str, traces: list[str]) -> str | None:
    """The policy the agent was actually given, as the first trace recorded it."""
    for name in traces:
        p = RUNS_DIR / run_id / "traces" / name
        if name and p.is_file():
            policy = json.loads(p.read_text()).get("policy")
            if policy:
                return str(policy)
    return None


def _no_db() -> str:
    """The one-line remedy, wherever the database is asked for and is not there."""
    return (
        "the central Postgres is not reachable: "
        "`make -C ../nmp-central-ai up`, then `make db-migrate` here"
    )


def _trial_number(trial: str) -> int:
    m = re.fullmatch(r"t?(\d+)", trial)
    if not m:
        raise HTTPException(404, "no such trial")
    return int(m.group(1))


def _mlflow_run_url(run_id: str | None) -> str | None:
    """A deep link into the central MLflow for one run. The viewer never reads the
    tracking server itself — the run folder is the record — so this is a link, not a fetch."""
    if not run_id:
        return None
    return f"{settings().mlflow_tracking_uri.rstrip('/')}/#/experiments/search?runId={run_id}"


def _json_or_none(path: Path) -> Any:
    """A committed JSON file's content, or None when it is absent or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ── a version's architecture, from its own folder (s13 §2) ─────────────────
@lru_cache(maxsize=32)
def _retrieval_info(variant: str) -> dict[str, Any]:
    """A retrieval variant's knowledge tools, dense model and policy template, read from tau2's
    variant spec once per process: no environment, sandbox or embedding model. Without tau2 (the
    demo image), or for a spec that will not load, only the name is known."""
    unknown: dict[str, Any] = {
        "variant": variant,
        "tools": [],
        "dense_model": None,
        "template": None,
    }
    if not replay.available():
        return unknown
    try:
        return retrieval_mod.variant_summary(variant)
    except Exception:  # noqa: BLE001 - the tab draws what it knows; a broken spec never 500s
        return unknown


def _cycles(domain: str) -> dict[str, dict[str, Any]]:
    """The ledger entry that made each challenger, by version name."""
    return {e["challenger"]: e for e in read_ledger(domain) if e.get("challenger")}


def _made_by(
    name: str, diag: dict[str, Any] | None, cycle: dict[str, Any] | None
) -> dict[str, Any]:
    """How a version was made and from what: the ledger's cycle, else its `diagnosis.json`."""
    if cycle and cycle.get("kind"):  # `make challenge`: a hand-made fork, gated like a challenger
        return {
            "kind": cycle["kind"],
            "from": cycle.get("forked_from") or cycle.get("champion"),
            "cycle": cycle.get("cycle"),
            "detail": "; ".join(cycle.get("agent_yaml") or []),
        }
    if cycle:
        opt = cycle.get("optimiser_model")
        mode = cycle.get("optimiser_mode")
        return {
            "kind": "loop cycle",
            "from": cycle.get("champion"),
            "cycle": cycle.get("cycle"),
            "detail": " ".join(x for x in (mode, opt, "optimiser") if x),
        }
    if diag and diag.get("kind"):
        return {
            "kind": diag["kind"],
            "from": diag.get("forked_from"),
            "cycle": None,
            "detail": "; ".join(diag.get("agent_yaml") or []),
        }
    if diag:
        return {"kind": "loop cycle", "from": None, "cycle": None, "detail": "no ledger entry"}
    return {
        "kind": "base",
        "from": None,
        "cycle": None,
        "detail": "tau2's instruction" if name == "v0" else "hand-built",
    }


def _architecture(
    v: AgentVersion, names: list[str], cycles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """What the Agent tab draws a version from without a run: its retrieval and the tools it
    gives, its surfaces (hashed, so a byte-for-byte copy reads as one), the layers of its system
    prompt in the order `compose()` joins them, how it was made, and the parent it is compared
    with: the version it was made from, else the one before it; none when its source is gone."""
    diag = _json_or_none(v.path / "diagnosis.json")
    made = _made_by(v.name, diag, cycles.get(v.name))
    if made["from"]:
        parent = made["from"] if made["from"] in names and made["from"] != v.name else None
    else:
        i = names.index(v.name) if v.name in names else 0
        parent = names[i - 1] if i > 0 else None
    info = _retrieval_info(v.retrieval) if v.retrieval else None
    files = {n: v.path / n for n in SURFACES if (v.path / n).is_file()}
    template = (info or {}).get("template")
    policy = (
        f"policy: {template}"
        if template
        else f"policy: {v.retrieval}'s template"
        if v.retrieval
        else "policy: tau2 domain policy"
    )
    layers = ["system.md", policy]
    if re.search(r"^def extra_context\b", v.helper or "", re.M):
        layers.append("extra_context(): helper.py")
    layers.append("CLOCK_NOTE")
    if v.config.identity_note:
        layers.append("identity note")
    return {
        "diagnosis": diag,
        "retrieval": v.retrieval,
        "retrieval_info": dict(info) if info else None,
        "tool_mode": v.config.tool_mode,
        "parallel_calls": v.config.parallel_calls,
        "surfaces_present": list(files),
        "surfaces": {
            n: {"sha": hashlib.sha256(p.read_bytes()).hexdigest()[:12], "chars": len(p.read_text())}
            for n, p in files.items()
        },
        "prompt_layers": layers,
        "made_by": made,
        "parent": parent,
    }


def _health(run_id: str) -> dict[str, Any] | None:
    """A run's harness health, computed once per version of its `tau2_results.json`."""
    try:
        mtime = (RUNS_DIR / run_id / "tau2_results.json").stat().st_mtime_ns
    except OSError:
        return None
    return _health_at(run_id, mtime)


@lru_cache(maxsize=16)
def _health_at(run_id: str, mtime_ns: int) -> dict[str, Any] | None:
    try:
        return run_health(run_id, RUNS_DIR)
    except Exception:  # noqa: BLE001 - a results file the arithmetic cannot read shows no health
        return None


def _conversation_health(run_id: str, task_id: str, trial: int) -> dict[str, Any] | None:
    h = _health(run_id)
    rows = (h or {}).get("conversations") or []
    return next(
        (r for r in rows if str(r.get("task_id")) == task_id and r.get("trial") == trial - 1),
        None,
    )


def _cut_of(m: RunMeta) -> int | None:
    """Which cut an older run's tasks are (its run.json predates `split_version`): 1 or 2, or None."""
    if m.split not in ("train", "test") or m.domain not in DOMAINS:
        return None
    try:
        s = read_split(m.domain)
    except FileNotFoundError:
        return None
    ids = set(m.task_ids)
    if ids and ids == set((s.get("v1") or {}).get(m.split) or []):
        return 1
    if ids and ids == set(s.get(m.split) or []):
        return int(s.get("version", 1))
    return None


def _tokens_per_conversation(results: list[TaskResult]) -> dict[str, int] | None:
    """The mean tokens one conversation spent: every role, and the agent's share of it.
    Read from `results.jsonl` rather than the summary in `run.json`, which has no token
    field and, the folder being scored, is never rewritten to gain one."""
    if not results:
        return None
    m = profile(results)["metrics"]
    return {
        "all": round(m["tokens"]["mean"]),
        "agent": round(m["agent_in"]["mean"] + m["agent_out"]["mean"]),
    }


def _mlflow_embeddable() -> bool:
    """Whether the tracking server answers at all, so a page can decide between an
    iframe and a plain link. Three seconds, and a failure is a `False`, never an error."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(settings().mlflow_tracking_uri, method="HEAD")  # noqa: S310
        with urllib.request.urlopen(req, timeout=3) as r:  # noqa: S310 - local platform URL
            return bool(200 <= r.status < 400)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _check_domain(domain: str) -> None:
    if domain not in DOMAINS and domain != SMOKE_DOMAIN:
        raise HTTPException(404, "no such domain")


def _which_split(task_id: str, split: dict[str, Any]) -> str:
    if task_id in split.get("train", []):
        return "train"
    if task_id in split.get("test", []):
        return "test"
    return "reserve"


def _task_spec(domain: str, task_id: str) -> dict[str, Any]:
    for t in read_task_extract(domain).get("tasks", []):
        if t.get("id") == task_id:
            return dict(t)
    raise HTTPException(404, "no such task")


def _task_row(t: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    desc = t.get("description") or {}
    ev = t.get("evaluation_criteria") or {}
    sc = (t.get("user_scenario") or {}).get("instructions") or {}
    return {
        "id": t.get("id"),
        "split": _which_split(str(t.get("id")), split),
        "purpose": desc.get("purpose"),
        "relevant_policies": desc.get("relevant_policies"),
        "reason_for_call": sc.get("reason_for_call") if isinstance(sc, dict) else None,
        # banking's purposes are tau2's placeholder (`Task: task_001`): the customer's goal in
        # one sentence, derived from the scenario (data/goals.py); None where the purpose is real
        "goal": customer_goal(sc) if is_placeholder(desc.get("purpose")) else None,
        "n_actions": len(ev.get("actions") or []),
        "n_communicate": len(ev.get("communicate_info") or []),
        "n_nl_assertions": len(ev.get("nl_assertions") or []),
        "n_env_assertions": len(ev.get("env_assertions") or []),
        "n_documents": len(t.get("required_documents") or []),
        "reward_basis": ev.get("reward_basis"),
    }


def _harness_tools(domain: str, extract_tools: list[dict[str, Any]]) -> dict[str, Any]:
    """The tools the domain's champion is given. The extract lists the tools of banking's default
    retrieval (`bm25_grep`: `KB_search`, `grep`); a champion on another variant gets that
    variant's knowledge tools in their place, read from tau2's spec. Without tau2 (the demo
    image) the variant's tools cannot be read, so the extract's are shown and `known` is false."""
    champ = (read_registry(domain).get("champion") or {}).get("agent")
    out: dict[str, Any] = {
        "version": champ,
        "retrieval": None,
        "retrieval_info": None,
        "extract_retrieval": BANKING_RETRIEVAL if domain == "banking_knowledge" else None,
        "replaced": [],
        "known": True,
        "tools": extract_tools,
    }
    try:
        v = load_version(domain, champ) if champ else None
    except FileNotFoundError:
        v = None
    if v is None or not v.retrieval:
        return out
    info = _retrieval_info(v.retrieval)
    out["retrieval"] = v.retrieval
    out["retrieval_info"] = dict(info)
    if v.retrieval == BANKING_RETRIEVAL:
        return out
    rows = _variant_tool_rows(v.retrieval)
    if not rows:
        out["known"] = False
        return out
    default = set(_retrieval_info(BANKING_RETRIEVAL).get("tools") or [])
    out["replaced"] = [t["name"] for t in extract_tools if t.get("name") in default]
    out["tools"] = [*rows, *(t for t in extract_tools if t.get("name") not in default)]
    return out


@lru_cache(maxsize=8)
def _variant_tool_rows_cached(variant: str) -> tuple[dict[str, Any], ...]:
    return tuple(retrieval_mod.variant_tool_rows(variant))


def _variant_tool_rows(variant: str) -> list[dict[str, Any]]:
    """A variant's knowledge tools with tau2's descriptions; [] without tau2 or for a spec that
    will not load, so the page shows what it knows rather than a 500."""
    if not replay.available():
        return []
    try:
        return [dict(r) for r in _variant_tool_rows_cached(variant)]
    except Exception:  # noqa: BLE001 - a broken spec never 500s the domain page
        return []


def _short(fraction: str | None) -> bool:
    """`action_checks` and friends are written `matched/total`; short of total is a miss."""
    if not fraction or "/" not in fraction:
        return False
    got, _, want = fraction.partition("/")
    try:
        return int(got) < int(want)
    except ValueError:
        return False


def _failure_mix(metas: list[Any]) -> dict[str, Any]:
    """Across every scored run, what actually failed in the conversations that failed.
    A conversation can miss on more than one check, so the counts do not sum to `failed`."""
    failed = 0
    mix = dict.fromkeys(("db_check", "actions", "communicate_info", "nl_assertions", "error"), 0)
    capped = 0
    for m in metas:
        if m.dry_run or not (m.summary or {}).get("n_scored"):
            continue
        try:
            _, results = load_run(m.run_id)
        except FileNotFoundError:
            continue
        for r in results:
            if r.correct is not False:
                continue
            failed += 1
            if r.db_check is False:
                mix["db_check"] += 1
            if _short(r.action_checks):
                mix["actions"] += 1
            if _short(r.communicate_checks):
                mix["communicate_info"] += 1
            if _short(r.nl_assertions):
                mix["nl_assertions"] += 1
            if r.error:
                mix["error"] += 1
            if r.termination_reason == "max_steps":
                capped += 1
    return {"failed": failed, "by_check": mix, "hit_turn_cap": capped}


def _basis_counts(ext: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in ext.get("tasks", []):
        for b in (t.get("evaluation_criteria") or {}).get("reward_basis") or []:
            out[str(b)] = out.get(str(b), 0) + 1
    return out


def trace_events(t: dict[str, Any]) -> list[dict[str, Any]]:
    """The conversation as a flat event list for the viewer."""
    events: list[dict[str, Any]] = []
    for i, m in enumerate(t.get("messages") or []):
        role = m.get("role")
        if role in {"user", "assistant"} and m.get("tool_calls"):
            for c in m["tool_calls"]:
                events.append(
                    {
                        "type": "tool_call",
                        "i": i,
                        "by": role,
                        "name": c.get("name"),
                        "arguments": c.get("arguments"),
                    }
                )
        elif role == "tool":
            events.append(
                {
                    "type": "tool_result",
                    "i": i,
                    "error": bool(m.get("error")),
                    "text": str(m.get("content") or ""),
                }
            )
        elif role in {"user", "assistant"}:
            events.append({"type": role, "i": i, "text": str(m.get("content") or "")})
    return events


def trace_messages(t: dict[str, Any]) -> list[dict[str, Any]]:
    """Every message whole, in order: what `trace_events` flattens away, kept.

    `id` pairs a tool result with the call that asked for it; `requestor` says whether
    the agent or the simulated customer made a call; `usage` and `seconds` are the cost
    of the model call that produced the message (absent on tau2's fixed greeting)."""
    out: list[dict[str, Any]] = []
    for i, m in enumerate(t.get("messages") or []):
        u = m.get("usage") or None
        out.append(
            {
                "i": i,
                "role": m.get("role"),
                "content": m.get("content"),
                "tool_calls": [
                    {
                        "id": c.get("id"),
                        "name": c.get("name"),
                        "arguments": c.get("arguments") or {},
                        "requestor": c.get("requestor") or m.get("role"),
                    }
                    for c in (m.get("tool_calls") or [])
                ],
                "id": m.get("id"),
                "requestor": m.get("requestor"),
                "turn_idx": m.get("turn_idx"),
                "usage": {
                    "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                }
                if u
                else None,
                "seconds": m.get("generation_time_seconds"),
                "error": bool(m.get("error")),
                # s09: what the version's code surfaces did on this reply — the reminder it rode
                # with, and a write a check blocked before this retry
                "harness": (m.get("raw_data") or {}).get("tau2_loop")
                if isinstance(m.get("raw_data"), dict)
                else None,
            }
        )
    return out


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA from this process when `frontend/dist` exists (absent in dev: Vite serves it)."""
    if not FRONTEND_DIST.is_dir():
        return
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    index = FRONTEND_DIST / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request) -> Any:
        if path.startswith("api/"):
            raise HTTPException(404)
        candidate = (FRONTEND_DIST / path).resolve()
        if path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)
