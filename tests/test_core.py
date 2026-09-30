"""The sealed core and the agent service, offline: what the SDK child can see, what the core
imports, and that a request answers the same in-process and over HTTP."""

from __future__ import annotations

import ast
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tau2_loop.llm import core

LLM_DIR = Path(core.__file__).parent
SHIPPED = ("core.py", "prompting.py", "service.py")  # what Dockerfile.agent copies

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_details",
            "description": "Look a user up.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    }
]
MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are an airline agent. Policy: be exact."},
    {"role": "user", "content": "Hi, I'm mia_li_3668 and I want to change a flight."},
]
REPLY = '{"content": null, "tool_calls": [{"name": "get_user_details", "arguments": {"user_id": "mia_li_3668"}}]}'


# ── the environment ──────────────────────────────────────────────────────
def test_sealed_env_keeps_the_allow_list_and_blanks_everything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BILLING", "subscription")
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "LC_ALL": "C",
        "CLAUDE_CODE_OAUTH_TOKEN": "tok",
        "DATABASE_URL": "postgresql://secret",
        "PG_SUPERUSER_URL": "postgresql://nmp:nmp@x/y",
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "ANTHROPIC_API_KEY": "sk-test",
        "CLAUDE_CODE_ENTRYPOINT": "cli",
        "CLAUDECODE": "1",
        "OPENAI_API_KEY": "sk-openai",
    }
    env = core.sealed_env(parent)
    # the SDK merges options.env over the parent's environment: this is what the child gets
    child = {**parent, **env}
    non_empty = {k for k, v in child.items() if v}
    assert non_empty == {
        "PATH",
        "HOME",
        "LC_ALL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    }
    assert child["DATABASE_URL"] == "" and child["ANTHROPIC_API_KEY"] == ""
    assert child["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"
    assert set(env) >= set(parent)  # every inherited name is overridden, not left to merge


def test_sealed_env_passes_the_key_only_when_billing_is_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BILLING", "api")
    assert core.sealed_env({"ANTHROPIC_API_KEY": "sk-x"})["ANTHROPIC_API_KEY"] == "sk-x"


def test_the_session_directory_is_outside_the_checkout() -> None:
    from tau2_loop.config import ROOT

    assert not core.session_dir().resolve().is_relative_to(ROOT.resolve())


# ── the import graph ────────────────────────────────────────────────────
def _our_imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tau2_loop"):
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith("tau2_loop")}
    return out


def test_the_shipped_files_import_nothing_of_ours_but_each_other() -> None:
    siblings = {"tau2_loop.llm.core", "tau2_loop.llm.prompting", "tau2_loop.llm"}
    for name in SHIPPED:
        assert _our_imports(LLM_DIR / name) <= siblings, name
    # `from tau2_loop.llm import core` is the one package import, and only for the module itself
    assert "tau2_loop.llm" not in _our_imports(LLM_DIR / "core.py")


def test_the_container_layout_imports_with_only_the_sdk_and_fastapi(tmp_path: Path) -> None:
    """The three files under empty package inits, as Dockerfile.agent lays them out."""
    pkg = tmp_path / "tau2_loop" / "llm"
    pkg.mkdir(parents=True)
    (tmp_path / "tau2_loop" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    for name in SHIPPED:
        shutil.copyfile(LLM_DIR / name, pkg / name)
    probe = (
        "import sys; import tau2_loop.llm.service as s; s.create_app('t'); "
        "bad = [m for m in ('litellm', 'tau2', 'tau2_loop.config', 'dotenv', 'mlflow') if m in sys.modules]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# ── one request, two routes ─────────────────────────────────────────────
class Stub:
    """core.run_query stand-in: records what the SDK would have been asked, returns a fixed reply."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | list[str], str, str]] = []

    def __call__(
        self, system: str, user: str | list[str], model: str, effort: str = "medium"
    ) -> core.SdkResult:
        self.calls.append((system, user, model, effort))
        return core.SdkResult(REPLY, 1234, 56, None, 7, "sess-1", cache_read=1100, cache_write=120)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Stub:
    s = Stub()
    monkeypatch.setattr(core, "run_query", s)
    import tau2_loop.llm.sdk_provider as provider

    monkeypatch.setattr(provider, "require_live", lambda: None)
    provider.register()
    return s


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def service_url() -> Any:
    import uvicorn

    from tau2_loop.llm.service import create_app

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app("secret-token"), host="127.0.0.1", port=port, log_level="error")
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)


def test_only_the_last_block_carries_a_cache_mark() -> None:
    """One mark, on this call's last turn: the next call reads back from it. The CLI uses
    three of the API's four marks, so a second one of ours is refused."""
    content = core.content_blocks(["ask", "[user]\nhi", "[assistant]\nId?", "[user]\nu1"])
    marked = [i for i, c in enumerate(content) if "cache_control" in c]
    assert marked == [3] and content[3]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in core.content_blocks(["ask"])[0]


def test_effort_reaches_the_in_process_provider(stub: Stub) -> None:
    import litellm

    litellm.completion(
        model="claude-sdk/claude-haiku-4-5",
        messages=MESSAGES,
        reasoning_effort="high",
        allowed_openai_params=["reasoning_effort"],
    )
    assert stub.calls[-1][2:] == ("claude-haiku-4-5", "high")


def test_one_request_answers_the_same_in_process_and_over_http(
    stub: Stub, service_url: str
) -> None:
    import litellm

    kw: dict[str, Any] = {
        "messages": MESSAGES,
        "tools": TOOLS,
        "tool_choice": "auto",
        "reasoning_effort": "medium",
        "allowed_openai_params": ["reasoning_effort"],
    }
    local = litellm.completion(model="claude-sdk/claude-sonnet-5", **kw)
    remote = litellm.completion(
        model="openai/claude-sonnet-5",
        api_base=f"{service_url}/v1",
        api_key="secret-token",
        num_retries=0,
        **kw,
    )
    # the SDK was asked the same thing, byte for byte
    assert len(stub.calls) == 2 and stub.calls[0] == stub.calls[1]
    assert stub.calls[0][2:] == ("claude-sonnet-5", "medium")

    def shape(r: Any) -> tuple[Any, ...]:
        m = r.choices[0].message
        calls = [(c.function.name, c.function.arguments) for c in m.tool_calls or []]
        return (
            m.content,
            calls,
            r.choices[0].finish_reason,
            r.usage.prompt_tokens,
            r.usage.completion_tokens,
            r.usage.prompt_tokens_details.cached_tokens,
            r.usage.prompt_tokens_details.cache_creation_tokens,
        )

    assert shape(local) == shape(remote)
    # the transcript went to the SDK as blocks, and the cache split came back on both routes
    assert isinstance(stub.calls[0][1], list) and len(stub.calls[0][1]) == len(MESSAGES)
    assert shape(remote)[-2:] == (1100, 120)
    assert shape(remote)[1] == [("get_user_details", '{"user_id": "mia_li_3668"}')]


def test_the_service_refuses_a_call_without_the_token_and_an_unserved_model(
    stub: Stub, service_url: str
) -> None:
    import httpx

    body = {"model": "claude-haiku-4-5", "messages": MESSAGES}
    url = f"{service_url}/v1/chat/completions"
    assert httpx.post(url, json=body).status_code == 401
    assert httpx.post(url, json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = httpx.post(url, json=body, headers={"Authorization": "Bearer secret-token"})
    assert ok.status_code == 200 and ok.json()["choices"][0]["message"]["role"] == "assistant"
    other = {**body, "model": "gpt-4o"}
    assert (
        httpx.post(url, json=other, headers={"Authorization": "Bearer secret-token"}).status_code
        == 400
    )
    health = httpx.get(f"{service_url}/healthz").json()
    assert health["ok"] and "claude-sonnet-5" in health["models"]
    assert stub.calls, "only the authorised call reached the core"
    assert len(stub.calls) == 1


def test_the_service_will_not_start_open(monkeypatch: pytest.MonkeyPatch) -> None:
    from tau2_loop.llm.service import create_app

    monkeypatch.delenv("AGENT_SERVICE_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        create_app()


# ── the harness's side of the route ─────────────────────────────────────
def test_the_runner_routes_the_agent_to_the_service_without_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tau2_loop.agent.versions import load_version
    from tau2_loop.eval import runner

    monkeypatch.setenv("AGENT_SERVICE_URL", "http://127.0.0.1:8090/")
    monkeypatch.setenv("AGENT_SERVICE_TOKEN", "secret-token")
    cfg = runner._run_config("airline", load_version("airline", "v0"), ["0"], 1, 1, 300)
    assert cfg.llm_agent == "openai/claude-haiku-4-5"
    assert cfg.llm_args_agent["api_base"] == "http://127.0.0.1:8090/v1"
    assert cfg.llm_args_agent["reasoning_effort"] == "medium"
    assert "secret-token" not in repr(cfg.model_dump())  # tau2 writes this config to disk
    assert cfg.llm_args_user["reasoning_effort"] == runner.USER_EFFORT
    assert runner.agent_route() == ("service:127.0.0.1:8090", "http://127.0.0.1:8090")
    monkeypatch.delenv("AGENT_SERVICE_URL")
    cfg = runner._run_config("airline", load_version("airline", "v0"), ["0"], 1, 1, 300)
    assert cfg.llm_agent == "claude-sdk/claude-haiku-4-5" and "api_base" not in cfg.llm_args_agent


def test_a_dry_run_records_effort_split_version_and_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.eval import runner

    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    monkeypatch.delenv("AGENT_SERVICE_URL", raising=False)
    meta, _ = runner.run_eval("airline", "v0", "train", dry_run=True, track=False)
    assert (meta.agent_effort, meta.user_effort) == ("medium", "medium")
    assert meta.split_version == 2 and meta.n_tasks == 25
    assert meta.agent_route == "in-process"
    monkeypatch.setenv("AGENT_SERVICE_URL", "http://127.0.0.1:8090")
    meta, _ = runner.run_eval("airline", "v0", "test", dry_run=True, track=False)
    assert meta.agent_route == "service:127.0.0.1:8090"
