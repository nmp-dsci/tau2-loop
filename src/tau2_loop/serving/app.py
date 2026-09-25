"""The API behind the frontend, and the frontend itself when a build is present.

Every route serves a committed file: `runs/`, `agents/`, `loop/`, `data/`.
There is no write route and no model call; the demo image is this process
with `DEMO_MODE=1` and nothing else.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tau2_loop import __version__
from tau2_loop.agent.versions import list_versions, load_version
from tau2_loop.config import DOMAINS, FRONTEND_DIST, RUNS_DIR, SMOKE_DOMAIN, settings
from tau2_loop.data.splits import read_split, read_task_extract
from tau2_loop.eval.compare import compare
from tau2_loop.eval.results import slug
from tau2_loop.eval.runner import list_runs, load_run
from tau2_loop.loop.ledger import read_ledger
from tau2_loop.tracking.registry import read_all, read_registry
from tau2_loop.tracking.snapshot import read_snapshot


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
        }

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
            "tasks": [_task_row(t, split) for t in ext.get("tasks", [])],
        }

    @app.get("/api/domains/{domain}/tasks/{task_id:path}")
    def task(domain: str, task_id: str) -> dict[str, Any]:
        _check_domain(domain)
        ext = read_task_extract(domain)
        for t in ext.get("tasks", []):
            if t.get("id") == task_id:
                return {**t, "split": _which_split(task_id, read_split(domain))}
        raise HTTPException(404, "no such task")

    # ── agents and runs ──────────────────────────────────────────────────
    @app.get("/api/agents")
    def agents(domain: str | None = None) -> dict[str, Any]:
        out = []
        for d in DOMAINS if domain is None else (domain,):
            for v in list_versions(d):
                diag = v.path / "diagnosis.json"
                out.append(
                    {
                        "domain": d,
                        "name": v.name,
                        "ref": v.ref,
                        "fingerprint": v.fingerprint,
                        "config": v.config.__dict__,
                        "has_helper": v.helper is not None,
                        "helper_functions": re.findall(r"^def (\w+)", v.helper or "", re.M),
                        "diagnosis": json.loads(diag.read_text()) if diag.exists() else None,
                        "runs": [m.run_id for m in list_runs(d) if m.agent == v.name],
                    }
                )
        return {"versions": out, "registry": read_all()}

    @app.get("/api/agents/diff")
    def agents_diff(domain: str, a: str, b: str) -> dict[str, Any]:
        """Unified diff of the two surfaces between versions, with the reasoning that produced `b`."""
        try:
            va, vb = load_version(domain, a), load_version(domain, b)
        except FileNotFoundError as e:
            raise HTTPException(404, "no such agent") from e
        fa, fb = va.files(), vb.files()
        files = []
        for name in ("system.md", "helper.py", "agent.yaml"):
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
        }

    @app.get("/api/runs")
    def runs(domain: str | None = None) -> list[dict[str, Any]]:
        return [m.__dict__ for m in list_runs(domain)]

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        try:
            meta, results = load_run(run_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "no such run") from e
        return {"meta": meta.__dict__, "results": [r.__dict__ for r in results]}

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

    @app.get("/api/runs/{run_id}/{task_id}/{trial}")
    def trial(run_id: str, task_id: str, trial: str) -> dict[str, Any]:
        """One conversation, addressed the way the viewer addresses it: task id and
        `t<n>`. The trace file name (`eval/runner.py` slugs the task id) stays on disk."""
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
        return {**trace(run_id, row.trace), "result": row.__dict__, "domain": meta.domain}

    @app.get("/api/compare")
    def compare_runs(a: str, b: str) -> dict[str, Any]:
        ma, ra = load_run(a)
        mb, rb = load_run(b)
        v = compare(ra, rb)
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
        return {"verdict": v.__dict__, "rows": rows, "a": ma.__dict__, "b": mb.__dict__}

    @app.get("/api/ledger")
    def ledger(domain: str | None = None) -> dict[str, list[dict[str, Any]]]:
        return {d: read_ledger(d) for d in (DOMAINS if domain is None else (domain,))}

    @app.get("/api/registry")
    def registry() -> dict[str, Any]:
        return read_all()

    @app.get("/api/experiments")
    def experiments() -> dict[str, Any]:
        return read_snapshot()

    _mount_frontend(app)
    return app


def _check_domain(domain: str) -> None:
    if domain not in DOMAINS and domain != SMOKE_DOMAIN:
        raise HTTPException(404, "no such domain")


def _which_split(task_id: str, split: dict[str, Any]) -> str:
    if task_id in split.get("train", []):
        return "train"
    if task_id in split.get("test", []):
        return "test"
    return "reserve"


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
        "n_actions": len(ev.get("actions") or []),
        "n_communicate": len(ev.get("communicate_info") or []),
        "n_nl_assertions": len(ev.get("nl_assertions") or []),
        "n_env_assertions": len(ev.get("env_assertions") or []),
        "reward_basis": ev.get("reward_basis"),
    }


def _basis_counts(ext: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in ext.get("tasks", []):
        for b in (t.get("evaluation_criteria") or {}).get("reward_basis") or []:
            out[str(b)] = out.get(str(b), 0) + 1
    return out


def trace_events(t: dict[str, Any]) -> list[dict[str, Any]]:
    """The conversation as a flat event list for the viewer."""
    events: list[dict[str, Any]] = []
    for m in t.get("messages") or []:
        role = m.get("role")
        if role in {"user", "assistant"} and m.get("tool_calls"):
            for c in m["tool_calls"]:
                events.append(
                    {
                        "type": "tool_call",
                        "by": role,
                        "name": c.get("name"),
                        "arguments": c.get("arguments"),
                    }
                )
        elif role == "tool":
            events.append(
                {
                    "type": "tool_result",
                    "error": bool(m.get("error")),
                    "text": str(m.get("content") or ""),
                }
            )
        elif role in {"user", "assistant"}:
            events.append({"type": role, "text": str(m.get("content") or "")})
    return events


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
