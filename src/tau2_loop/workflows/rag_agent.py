"""workflow_rag's RAG agent (s16): one agentic-RAG session per train question, live in the viewer.

A session reads one customer's train script, researches the knowledge base with the answering
agent's own knowledge tools (v4's `alltools_minilm`: BM25, dense search and the read-only shell)
and writes the workflow the answering agent would follow, as JSON. The model runs on the
subscription through `llm.core.answer`, one SDK session per turn with native tool calls, exactly
as the answering agent does; this module runs each call in tau2's banking environment and feeds
the result back, until the model writes the workflow or reaches the version's step limit.

What it never sees: gold. Its prompt is the version's `rag_agent.md`, the answering agent's
policy and tool schemas, and the customer's script. Gold (the task's actions and required
documents) is read only by `score()`, after the session, to show how close it came. Only train
questions can be started; a test id is refused before anything runs. The shell is tau2's
best-effort sandbox, so a command that names the task files is refused here as well.

A version lives in `rag_agents/<domain>/<rN>/` (prompt + frozen settings, like `agents/`); each
session in `rag_agent_runs/<id>/` (`run.json`, `events.jsonl`, `output.json`, `score.json`), kept
like a run folder: never edited once it ends.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tau2_loop.config import ROOT
from tau2_loop.data import documents as kb
from tau2_loop.data.splits import read_task_extract, split_ids

RAG_AGENTS_DIR = ROOT / "rag_agents"
# what precedes the question in the agent's first message: a train script, or a live request
SCRIPT_OPENING = "The customer's situation, as their script tells it:"
REQUEST_OPENING = (
    "An answering agent is talking to a customer now and asks for the procedure. Its request, in "
    "its own words (you do not see the customer):"
)
RAG_RUNS_DIR = ROOT / "rag_agent_runs"
QUOTE_CHARS = 300
# a shell command may not reach for the task files, whatever tau2's sandbox allows
SHELL_REFUSED = re.compile(r"task|evaluation_criteria|\.\.|~|^\s*/|\s/", re.I)
DOC_ID = re.compile(r"\bdoc_[A-Za-z0-9_()\-]+")
# how a quote cites the answering agent's policy rather than a knowledge-base document
POLICY_REFS = frozenset({"policy", "the policy", "agent policy", "customer service policy"})


class RagAgentError(ValueError):
    """A request a session cannot serve: an unknown version, a non-train question."""


@dataclass(frozen=True)
class RagAgent:
    name: str
    domain: str
    model: str
    effort: str
    retrieval: str
    tool_mode: str
    max_steps: int
    result_chars: int
    prompt: str  # rag_agent.md, unformatted


def versions(domain: str) -> list[str]:
    root = RAG_AGENTS_DIR / domain
    if not root.is_dir():
        return []
    return sorted(
        (p.name for p in root.iterdir() if (p / "rag_agent.yaml").is_file()),
        key=lambda n: int(re.sub(r"\D", "", n) or 0),
    )


def load(domain: str, name: str) -> RagAgent:
    d = RAG_AGENTS_DIR / domain / name
    if not re.fullmatch(r"r\d+", name) or not (d / "rag_agent.yaml").is_file():
        raise RagAgentError(f"no RAG agent {name} for {domain}")
    cfg = yaml.safe_load((d / "rag_agent.yaml").read_text())
    return RagAgent(
        name=name,
        domain=domain,
        model=str(cfg["model"]),
        effort=str(cfg["effort"]),
        retrieval=str(cfg["retrieval"]),
        tool_mode=str(cfg.get("tool_mode", "native_parallel")),
        max_steps=int(cfg.get("max_steps", 30)),
        result_chars=int(cfg.get("result_chars", 20000)),
        prompt=(d / "rag_agent.md").read_text(),
    )


def train_question(domain: str, task_id: str) -> str:
    """The customer's whole train script; a test or unknown id is refused."""
    if task_id not in set(split_ids(domain, "train")):
        raise RagAgentError(f"{task_id} is not a train question: test is sealed")
    for t in read_task_extract(domain)["tasks"]:
        if t["id"] == task_id:
            return str(((t.get("user_scenario") or {}).get("instructions")) or "").strip()
    raise RagAgentError(f"{task_id} is not in the task extract")


# ── the environment: tau2's banking tools, built once per process ──

_env_lock = threading.Lock()
_envs: dict[str, Any] = {}


def environment(retrieval: str) -> Any:
    """tau2's banking environment with the given retrieval variant, built once and shared:
    sessions call only its read-only knowledge tools."""
    with _env_lock:
        env = _envs.get(retrieval)
        if env is None:
            from tau2_loop.config import quiet_tau2

            quiet_tau2()
            from tau2_loop.eval.retrieval import register
            from tau2_loop.llm import scrub_injected_key

            register()
            from tau2.domains.banking_knowledge.environment import get_environment

            scrub_injected_key()  # tau2's dotenv search may have pulled a key in
            env = get_environment(retrieval_variant=retrieval)
            _envs[retrieval] = env
        return env


def _schema(t: Any) -> dict[str, Any]:
    return dict(t.openai_schema)


def tool_lists(env: Any, retrieval: str) -> tuple[list[dict[str, Any]], str, str]:
    """(the knowledge tools the agent calls, the answering agent's other tools as text, the
    customer's tools as text)."""
    from tau2_loop.eval.retrieval import variant_tools

    kb_names = set(variant_tools(retrieval))
    callable_ = [_schema(t) for t in env.get_tools() if t.name in kb_names]

    def listed(tools: list[Any]) -> str:
        rows = []
        for t in tools:
            fn = _schema(t)["function"]
            params = ", ".join((fn.get("parameters") or {}).get("properties", {}))
            first = ((fn.get("description") or "").strip().splitlines() or [""])[0]
            rows.append(f"- `{fn['name']}({params})`: {first}")
        return "\n".join(rows)

    agent = [t for t in env.get_tools() if t.name not in kb_names]
    return callable_, listed(agent), listed(list(env.get_user_tools()))


def system_prompt(agent: RagAgent, env: Any) -> tuple[str, list[dict[str, Any]]]:
    tools, agent_tools, user_tools = tool_lists(env, agent.retrieval)
    text = agent.prompt.format(policy=env.policy, agent_tools=agent_tools, user_tools=user_tools)
    return text, tools


# ── pure helpers, unit-tested ──


def shell_refusal(command: str) -> str | None:
    """Why a shell command is refused, or None. The knowledge base folder is flat, so nothing
    a search needs reaches outside it."""
    m = SHELL_REFUSED.search(command)
    if m:
        return (
            f"Refused: the command contains {m.group(0)!r}. Commands run inside the knowledge "
            "base folder only, on its documents."
        )
    return None


def parse_workflow(text: str | None) -> dict[str, Any] | None:
    """The JSON object in a final reply: the whole text, else the outermost {...}."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    for cand in (s, s[s.find("{") : s.rfind("}") + 1] if "{" in s else ""):
        try:
            v = json.loads(cand)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict) and isinstance(v.get("jobs"), list):
            return v
    return None


def _norm_id(s: str) -> str:
    return re.sub(r"[()]", "_", s.strip()).removesuffix(".md")


def resolve_doc(domain: str, ref: str) -> str | None:
    """A cited document as its id: the id itself, or the shell's file name for it."""
    index = kb.read_index(domain)
    if ref in index:
        return ref
    want = _norm_id(ref)
    return next((i for i in index if _norm_id(i) == want), None)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("‑", "-")).strip()


def quotes(o: Any) -> list[dict[str, str]]:
    """Every {quote, doc} pair anywhere in a workflow."""
    out: list[dict[str, str]] = []
    if isinstance(o, dict):
        if isinstance(o.get("quote"), str) and o.get("quote"):
            out.append({"quote": o["quote"], "doc": str(o.get("doc") or "")})
        for v in o.values():
            out.extend(quotes(v))
    elif isinstance(o, list):
        for v in o:
            out.extend(quotes(v))
    return out


def quote_check(domain: str, wf: dict[str, Any], policy: str = "") -> dict[str, Any]:
    """Each quote looked up in the document it cites, whitespace folded; no model. A quote that
    cites the policy (not a knowledge-base document) is looked up in the policy the agent read."""
    rows = []
    for q in quotes(wf):
        if q["doc"].strip().lower() in POLICY_REFS:
            doc_id: str | None = "policy"
            text = policy or None
        else:
            doc_id = resolve_doc(domain, q["doc"]) if q["doc"] else None
            d = kb.document(domain, doc_id) if doc_id else None
            text = d["content"] if d else None
        found = text is not None and _squash(q["quote"].strip(" .…\"'")) in _squash(text)
        rows.append({**q, "doc_id": doc_id, "found": found, "long": len(q["quote"]) > QUOTE_CHARS})
    return {"n": len(rows), "found": sum(r["found"] for r in rows), "rows": rows}


def cited_docs(domain: str, wf: dict[str, Any]) -> list[str]:
    refs = [str(d.get("id") or "") for d in wf.get("documents") or [] if isinstance(d, dict)]
    refs += [q["doc"] for q in quotes(wf)]
    out: list[str] = []
    for r in refs:
        i = resolve_doc(domain, r) if r else None
        if i and i not in out:
            out.append(i)
    return out


def seen_docs(domain: str, events: list[dict[str, Any]]) -> list[str]:
    """Documents any tool result showed: a search hit's id, or a file name the shell printed."""
    out: list[str] = []
    for e in events:
        if e.get("kind") != "tool":
            continue
        for ref in DOC_ID.findall(str(e.get("content") or "")):
            i = resolve_doc(domain, ref)
            if i and i not in out:
                out.append(i)
    return out


def workflow_tools(wf: dict[str, Any]) -> list[str]:
    """The tools a workflow's steps call, a discoverable one by its own name."""
    out: list[str] = []
    for job in wf.get("jobs") or []:
        for st in job.get("steps") or [] if isinstance(job, dict) else []:
            c = st.get("call") if isinstance(st, dict) else None
            if isinstance(c, str) and c and c not in out:
                out.append(c)
    return out


def _gold_tool(name: str) -> str:
    return name.removeprefix("give ")


def score(
    domain: str,
    task_id: str,
    wf: dict[str, Any] | None,
    events: list[dict[str, Any]],
    policy: str = "",
) -> dict[str, Any]:
    """How close the session came to the golden entry: read only here, after the session."""
    from tau2_loop.workflows.golden import golden_set

    gs = golden_set(domain) or {"entries": []}
    entry = next((e for e in gs["entries"] if e["task_id"] == task_id), None)
    required = [d["id"] for d in (entry or {}).get("required_documents", [])]
    seen = seen_docs(domain, events)
    out: dict[str, Any] = {
        "golden": None
        if entry is None
        else {
            "workflows": entry["workflows"],
            "info": entry["info"],
            "calls": entry["calls"],
            "required_documents": entry["required_documents"],
        },
        "documents_seen": {
            "required": len(required),
            "hit": [d for d in required if d in seen],
            "n": len(seen),
        },
    }
    if wf is None:
        return out
    cited = cited_docs(domain, wf)
    gold_tools = sorted({_gold_tool(c["tool"]) for c in (entry or {}).get("calls", [])})
    tools = workflow_tools(wf)
    out.update(
        {
            "documents_referenced": {
                "required": len(required),
                "hit": [d for d in required if d in cited],
                "extra": [d for d in cited if d not in required],
            },
            "quotes": quote_check(domain, wf, policy),
            "tools": {
                "gold": gold_tools,
                "hit": [t for t in gold_tools if t in tools],
                "extra": [t for t in tools if t not in gold_tools],
            },
            "jobs": [str(j.get("job")) for j in wf.get("jobs") or [] if isinstance(j, dict)],
        }
    )
    return out


# ── sessions ──


@dataclass
class SessionMeta:
    id: str
    domain: str
    agent: str
    task_id: str
    model: str
    effort: str
    retrieval: str
    started: str
    status: str = "running"  # running | done | failed | interrupted
    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None
    # "question": started on a train question (the demo); "conversation": an answering agent's
    # `request_workflow` mid-conversation, which reads only the agent's words and is never cached
    source: str = "question"
    conversation: str | None = None  # the live conversation that asked, when one did


_live: dict[str, SessionMeta] = {}
_live_lock = threading.Lock()


def session_dir(session_id: str) -> Path:
    if not re.fullmatch(r"[0-9TZ]+_[a-z_]+_r\d+_(task_\d+|req_[0-9a-f]{6})", session_id):
        raise RagAgentError(f"no session {session_id}")
    return RAG_RUNS_DIR / session_id


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_meta(d: Path, meta: SessionMeta) -> None:
    (d / "run.json").write_text(json.dumps(asdict(meta), indent=1) + "\n")


Ask = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None, str, str, str], Any]
RunTool = Callable[[str, dict[str, Any]], str]


def default_ask(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    effort: str,
    tool_mode: str,
) -> Any:
    from tau2_loop.llm import core

    return core.answer(messages, tools, model, effort=effort, tool_mode=tool_mode)


def env_runner(env: Any) -> RunTool:
    from tau2.data_model.message import ToolCall

    lock = threading.Lock()

    def run(name: str, args: dict[str, Any]) -> str:
        with lock:
            msg = env.get_response(
                ToolCall(id="rag", name=name, arguments=args, requestor="assistant")
            )
        return str(msg.content or "")

    return run


def run_session(
    d: Path,
    meta: SessionMeta,
    agent: RagAgent,
    system: str,
    tools: list[dict[str, Any]],
    question: str,
    ask: Ask,
    run_tool: RunTool,
    policy: str = "",
    opening: str = SCRIPT_OPENING,
) -> SessionMeta:
    """The agentic loop: ask, run every tool call, feed the results back, until the model writes
    the workflow. Every step is appended to `events.jsonl` as it happens, which the viewer polls."""
    events_path = d / "events.jsonl"
    events: list[dict[str, Any]] = []
    started = time.time()

    def emit(row: dict[str, Any]) -> None:
        row = {"t": int((time.time() - started) * 1000), **row}
        events.append(row)
        _append(events_path, row)

    names = {t["function"]["name"] for t in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"{opening}\n\n{question}",
        },
    ]
    emit({"kind": "start", "question": question})
    wf: dict[str, Any] | None = None
    nudged = retried = False
    try:
        while meta.steps < agent.max_steps + 3:
            over = meta.steps >= agent.max_steps
            if over and not nudged:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your research steps are used up. Write the workflow JSON now from what you have.",
                    }
                )
                nudged = True
            ans = ask(messages, None if over else tools, meta.model, meta.effort, agent.tool_mode)
            meta.steps += 1
            res = getattr(ans, "result", None)
            if res is not None:
                meta.input_tokens += int(res.input_tokens or 0)
                meta.output_tokens += int(res.output_tokens or 0)
            emit(
                {
                    "kind": "model",
                    "step": meta.steps,
                    "text": ans.content,
                    "calls": [
                        {"id": c["id"], "name": c["name"], "args": c["arguments"]}
                        for c in ans.tool_calls
                    ],
                    "input_tokens": getattr(res, "input_tokens", None),
                    "output_tokens": getattr(res, "output_tokens", None),
                    "cache_read": getattr(res, "cache_read", None),
                    "ms": getattr(res, "duration_ms", None),
                }
            )
            if ans.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {
                                    "name": c["name"],
                                    "arguments": json.dumps(c["arguments"]),
                                },
                            }
                            for c in ans.tool_calls
                        ],
                    }
                )
                for c in ans.tool_calls:
                    args = c["arguments"] if isinstance(c["arguments"], dict) else {}
                    refused = None
                    if c["name"] not in names:
                        refused = f"Refused: {c['name']} is not one of your tools."
                    elif c["name"] == "shell":
                        refused = shell_refusal(str(args.get("command", "")))
                    out = refused or run_tool(c["name"], args)
                    cut = len(out) > agent.result_chars
                    if cut:
                        out = (
                            out[: agent.result_chars]
                            + f"\n[cut at {agent.result_chars:,} of {len(out):,} characters]"
                        )
                    meta.tool_calls += 1
                    emit(
                        {
                            "kind": "tool",
                            "step": meta.steps,
                            "id": c["id"],
                            "name": c["name"],
                            "args": args,
                            "content": out,
                            "refused": bool(refused),
                            "cut": cut,
                        }
                    )
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
                _write_meta(d, meta)
                continue
            messages.append({"role": "assistant", "content": ans.content or ""})
            wf = parse_workflow(ans.content)
            if wf is not None or retried:
                break
            retried = True  # one more chance to write it as JSON
            emit({"kind": "note", "text": "The reply was not the workflow JSON; asked once more."})
            messages.append(
                {
                    "role": "user",
                    "content": "Reply with the workflow as one JSON object only, in the shape the instructions give.",
                }
            )
        meta.status = "done" if wf is not None else "failed"
        if wf is None:
            meta.error = "no workflow JSON in the final reply"
    except Exception as e:  # noqa: BLE001 - a failed session is recorded, not raised into the server
        meta.status = "failed"
        meta.error = f"{type(e).__name__}: {e}"[:500]
        emit({"kind": "error", "text": meta.error, "trace": traceback.format_exc()[-2000:]})
    meta.duration_ms = int((time.time() - started) * 1000)
    if wf is not None:
        (d / "output.json").write_text(json.dumps(wf, indent=1, ensure_ascii=False) + "\n")
    try:
        sc = score(meta.domain, meta.task_id, wf, events, policy)
        (d / "score.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        emit({"kind": "error", "text": f"scoring failed: {type(e).__name__}: {e}"[:500]})
    emit({"kind": "done", "status": meta.status})
    _write_meta(d, meta)
    return meta


def _launch(
    domain: str,
    agent: RagAgent,
    session_tail: str,
    question: str,
    opening: str,
    model: str | None,
    effort: str | None,
    *,
    task_id: str = "",
    source: str = "question",
    conversation: str | None = None,
    ask: Ask | None,
    run_tool: RunTool | None,
    env: Any,
    background: bool,
    on_start: Callable[[SessionMeta], None] | None = None,
) -> SessionMeta:
    from tau2_loop.llm import core

    model = model or agent.model
    effort = effort or agent.effort
    if effort not in core.EFFORTS:
        raise RagAgentError(f"effort is one of {', '.join(core.EFFORTS)}")
    if core.resolve_model(model) not in core.MODELS.values():
        raise RagAgentError(f"model is one of {', '.join(core.MODELS)}")
    if ask is None:
        from tau2_loop.llm import require_live

        require_live()  # refuses the demo image and a key that would bill per token
    env = env if env is not None else environment(agent.retrieval)
    system, tools = system_prompt(agent, env)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = SessionMeta(
        id=f"{stamp}_{domain}_{agent.name}_{session_tail}",
        domain=domain,
        agent=agent.name,
        task_id=task_id,
        model=core.resolve_model(model),
        effort=effort,
        retrieval=agent.retrieval,
        started=stamp,
        source=source,
        conversation=conversation,
    )
    d = session_dir(meta.id)
    d.mkdir(parents=True, exist_ok=False)
    (d / "system.md").write_text(system)
    _write_meta(d, meta)
    with _live_lock:
        _live[meta.id] = meta
    if on_start is not None:
        on_start(meta)
    args = (
        d,
        meta,
        agent,
        system,
        tools,
        question,
        ask or default_ask,
        run_tool or env_runner(env),
        str(getattr(env, "policy", "") or ""),
        opening,
    )
    if not background:
        return run_session(*args)
    threading.Thread(target=run_session, args=args, daemon=True, name=f"rag-{meta.id}").start()
    return meta


def start(
    domain: str,
    task_id: str,
    agent_name: str = "r1",
    model: str | None = None,
    effort: str | None = None,
    *,
    ask: Ask | None = None,
    run_tool: RunTool | None = None,
    env: Any = None,
    background: bool = True,
) -> SessionMeta:
    """Start a session on a train question; returns at once, the session running in a thread."""
    agent = load(domain, agent_name)
    question = train_question(domain, task_id)
    return _launch(
        domain,
        agent,
        task_id,
        question,
        SCRIPT_OPENING,
        model,
        effort,
        task_id=task_id,
        ask=ask,
        run_tool=run_tool,
        env=env,
        background=background,
    )


def start_request(
    domain: str,
    request: str,
    agent_name: str = "r1",
    *,
    conversation: str | None = None,
    on_start: Callable[[SessionMeta], None] | None = None,
    ask: Ask | None = None,
    run_tool: RunTool | None = None,
    env: Any = None,
) -> SessionMeta:
    """Research an answering agent's request, mid-conversation, and return when it is written.

    It reads only the request (the agent's words), never a task's script or gold, and its
    workflow is not cached into the library: the conversation could be a test one."""
    import secrets

    agent = load(domain, agent_name)
    text = request.strip()[:4000]
    if not text:
        raise RagAgentError("an empty request")
    return _launch(
        domain,
        agent,
        f"req_{secrets.token_hex(3)}",
        text,
        REQUEST_OPENING,
        None,
        None,
        source="conversation",
        conversation=conversation,
        ask=ask,
        run_tool=run_tool,
        env=env,
        background=False,
        on_start=on_start,
    )


def _read_meta(d: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((d / "run.json").read_text())
    with _live_lock:
        live = meta["id"] in _live
    if meta["status"] == "running" and not live:
        meta["status"] = "interrupted"  # the server restarted under it
    return meta


def sessions(domain: str) -> list[dict[str, Any]]:
    if not RAG_RUNS_DIR.is_dir():
        return []
    out = []
    for d in sorted(RAG_RUNS_DIR.iterdir(), reverse=True):
        if (d / "run.json").is_file() and f"_{domain}_" in d.name:
            out.append(_read_meta(d))
    return out


def session(domain: str, session_id: str) -> dict[str, Any]:
    d = session_dir(session_id)
    if not (d / "run.json").is_file() or f"_{domain}_" not in session_id:
        raise RagAgentError(f"no session {session_id}")
    events = []
    if (d / "events.jsonl").is_file():
        events = [
            json.loads(line)
            for line in (d / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]

    def opt(name: str) -> Any:
        p = d / name
        return json.loads(p.read_text()) if p.is_file() else None

    return {
        "meta": _read_meta(d),
        "events": events,
        "output": opt("output.json"),
        "score": opt("score.json"),
    }


def describe(domain: str, name: str) -> dict[str, Any]:
    """A version as the Agent tab shows it: settings, prompt and the tools it calls."""
    from tau2_loop.eval.retrieval import variant_tool_rows

    agent = load(domain, name)
    return {
        "name": agent.name,
        "model": agent.model,
        "effort": agent.effort,
        "retrieval": agent.retrieval,
        "tool_mode": agent.tool_mode,
        "max_steps": agent.max_steps,
        "result_chars": agent.result_chars,
        "prompt": agent.prompt,
        "tools": variant_tool_rows(agent.retrieval),
    }
