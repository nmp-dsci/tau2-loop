"""An optimiser per dataset (s13 §5): `optimisers/<domain>/`, today's optimiser as every default.

Each default profile reproduces today's prompts byte for byte (pinned below for all four
datasets); a guide reaches only its own dataset's prompts, under its own heading; a profile's
model, effort, turn limit, budgets and surfaces reach the sessions; a dataset's guards refuse
what its optimiser wrote. No model is called: the SDK client is a fake.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import tau2_loop.loop.optimiser as opt
import tau2_loop.loop.profiles as profiles
from tau2_loop.agent.versions import AgentVersion
from tau2_loop.config import DOMAINS
from tau2_loop.eval.results import TaskResult
from tau2_loop.llm import EFFORT, resolve_model
from tau2_loop.loop.guards import profile_violations
from tau2_loop.loop.profiles import OptimiserProfile, load_profile

# datasets whose committed profile is tuned, so not today's default: banking since s14 (routing,
# and the lessons in its guide.md), checked by its own test at the end
TUNED = ("banking_knowledge",)


def _tmp_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "optimisers"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(profiles, "OPTIMISERS_DIR", root)
    return root


def _write_profile(root: Path, domain: str, **files: str) -> Path:
    """`profile_yaml=`, `guide_md=`, `guards_py=` → optimisers/<domain>/{profile.yaml, …}."""
    d = root / domain
    d.mkdir(parents=True, exist_ok=True)
    for key, text in files.items():
        (d / key.replace("_", ".")).write_text(text)
    return d


def _champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str) -> AgentVersion:
    """A fixed champion under tmp_path: the same bytes before and after any change."""
    import tau2_loop.agent.versions as versions

    agents = tmp_path / "agents"
    monkeypatch.setattr(versions, "AGENTS_DIR", agents)
    monkeypatch.setattr(opt, "AGENTS_DIR", agents)
    d = agents / domain / "v3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "system.md").write_text(f"You are a {domain} agent.\n{{policy}}\n")
    (d / "helper.py").write_text("def on_reply(text):\n    return text\n")
    (d / "agent.yaml").write_text("model: sonnet\neffort: medium\n")
    return versions.load_version(domain, "v3")


def _context(domain: str, profile: OptimiserProfile | None) -> opt.Context:
    """A fixed context: what the prompts read of a cycle, without reading a run."""
    return opt.Context(
        domain=domain,
        blocks=["## Task 1 (trial 1)\nfailed", "## Task 2 (trial 1)\nfailed too"],
        history="Earlier cycles: none",
        held_block="none",
        fork_note=" a fork note.",
        tool_names="a, b, c",
        read_traces=[],
        halves=None,
        record="| task | v3 |",
        profile=profile,
    )


def _prompts(champion: AgentVersion, ctx: opt.Context) -> dict[str, str]:
    """The three prompts an optimiser can get: classic, routing's diagnosis, routing's writer."""
    return {
        "classic": opt.build_prompt(champion, "v9", "run0", [], ctx),
        "diagnose": opt.build_diagnose_prompt(champion, "v9", [], ctx),
        "write": opt.build_write_prompt(
            champion, "v9", [], ctx, {"surfaces": {"checks.py": "x"}}, ["checks.py"]
        ),
    }


def _fake_sdk(
    monkeypatch: pytest.MonkeyPatch, on_query: Callable[[str], None]
) -> list[dict[str, Any]]:
    """The Agent SDK client, faked: each session's options and prompt, and `on_query` acting as
    the session (writing files) instead of a model."""
    sessions: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, options: Any) -> None:
            sessions.append({"options": options})

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            sessions[-1]["prompt"] = prompt
            on_query(prompt)

        async def receive_response(self) -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(opt, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(opt, "require_live", lambda: None)
    monkeypatch.setattr(opt, "subscription_env", lambda: {})
    return sessions


def _deny(guard: Any, path: Path) -> bool:
    out = asyncio.run(guard({"tool_input": {"file_path": str(path)}}, None, None))
    return bool(out) and out["hookSpecificOutput"]["permissionDecision"] == "deny"


# ── the defaults: today's optimiser ──────────────────────────────────────────


def test_every_default_is_todays_optimiser_and_every_dataset_has_one() -> None:
    """The values the one optimiser ran with before s13, and the four committed profiles hold
    exactly those, with an empty guide and no guards."""
    base = OptimiserProfile("airline")
    assert (base.model, base.effort, base.mode, base.max_turns) == (
        "opus",
        "medium",
        "classic",
        120,
    )
    assert base.effort == EFFORT  # the effort every optimiser session ran at
    assert (base.transcript_budget, base.scenario_budget, base.diff_budget) == (
        60_000,
        60_000,
        90_000,
    )
    assert base.surfaces == {
        "classic": ("system.md", "helper.py"),
        "routing": ("system.md", "helper.py", "checks.py", "memory.py", "guidance.py"),
    }
    # the names the optimiser and its tests have always used are the profile's defaults
    assert (opt.MAX_TURNS, opt.TRANSCRIPT_BUDGET, opt.SCENARIO_BUDGET, opt.DIFF_BUDGET) == (
        120,
        60_000,
        60_000,
        90_000,
    )
    assert opt.MODES == ("classic", "routing")
    for d in DOMAINS:
        p = load_profile(d)
        assert (profiles.profile_dir(d) / "profile.yaml").is_file(), d
        if d in TUNED:
            continue
        assert (profiles.profile_dir(d) / "guide.md").read_text() == "", d
        assert p.is_default() and p.guide == "" and p.guards is None, d
        assert dataclasses.replace(p, fingerprint=base.fingerprint) == dataclasses.replace(
            base, domain=d
        )
        assert p.fingerprint == profiles.fingerprint(d) != profiles.EMPTY_FINGERPRINT


def test_a_dataset_with_no_profile_files_runs_todays_optimiser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tmp_profiles(tmp_path, monkeypatch)
    assert load_profile("airline") == OptimiserProfile("airline")
    assert profiles.fingerprint("airline") == profiles.EMPTY_FINGERPRINT
    # a profile.yaml that sets one key keeps today's value for every other
    _write_profile(root, "retail", profile_yaml="max_turns: 40\n")
    p = load_profile("retail")
    assert p.max_turns == 40 and p.model == "opus" and p.diff_budget == 90_000
    assert not p.is_default()


def test_the_four_default_profiles_build_todays_prompts_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For each dataset, from fixed inputs: the classic, diagnosis and writer prompts built with
    no profile at all (the template as it was before s13), with the code's defaults, and with
    the committed `optimisers/<domain>/` are one and the same string."""
    for d in DOMAINS:
        champion = _champion(tmp_path, monkeypatch, d)
        before = _prompts(champion, _context(d, None))
        committed = () if d in TUNED else (load_profile(d),)
        for profile in (OptimiserProfile(d), *committed):
            assert _prompts(champion, _context(d, profile)) == before, d
        assert all("# This dataset's lessons" not in p for p in before.values())
        assert all("optimiser profile" not in p for p in before.values())


def test_the_default_budgets_cut_the_failures_as_before_and_a_profile_recuts_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_context` shares the profile's three budgets out over the failures: today's 60,000 /
    60,000 / 90,000 by default; a dataset's own when its profile sets them."""
    champion = _champion(tmp_path, monkeypatch, "mock")
    limits: dict[str, list[int | None]] = {"trace": [], "goal": [], "diff": []}

    def task_block(domain: str, task_id: str, goal_chars: int | None = None, **_: Any) -> str:
        limits["goal"].append(goal_chars)
        return "PURPOSE: x"

    def diff_block(run_id: str, r: TaskResult, domain: str, limit: int | None = None) -> str:
        limits["diff"].append(limit)
        return "DATABASE DIFFERENCE"

    def condense(path: Path, limit: int = 0) -> str:
        limits["trace"].append(limit)
        return "### agent\nhi"

    monkeypatch.setattr(opt, "task_block", task_block)
    monkeypatch.setattr(opt, "diff_block", diff_block)
    monkeypatch.setattr(opt, "condense_trace", condense)
    monkeypatch.setattr(opt, "failure_details", lambda *a, **k: "reward 0.0")
    monkeypatch.setattr(opt, "halves", lambda d: None)
    failures = [TaskResult(str(i), 1, 0.0, False, trace=f"{i}.json") for i in range(10)]

    ctx = opt.build_context(champion, "run0", failures, OptimiserProfile("mock"))
    assert ctx.profile == OptimiserProfile("mock")
    assert set(limits["trace"]) == {6000} and set(limits["goal"]) == {6000}
    assert set(limits["diff"]) == {9000}
    for v in limits.values():
        v.clear()
    tuned = OptimiserProfile(
        "mock", transcript_budget=20_000, scenario_budget=30_000, diff_budget=15_000
    )
    opt.build_context(champion, "run0", failures, tuned)
    assert set(limits["trace"]) == {2000} and set(limits["goal"]) == {3000}
    assert set(limits["diff"]) == {1500}


# ── a dataset's guide ────────────────────────────────────────────────────────


def test_a_guide_reaches_only_its_own_dataset_after_the_shared_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Banking's guide is appended, under its own heading, to banking's three prompts and to no
    other dataset's; the shared prompt before it is unchanged. A blank guide adds nothing."""
    root = _tmp_profiles(tmp_path, monkeypatch)
    guide = "1. Research first: search the customer's stated problem before any generic gate.\n"
    _write_profile(root, "banking_knowledge", guide_md="\n" + guide + "\n\n")
    _write_profile(root, "retail", guide_md="  \n\n")
    for d in DOMAINS:
        champion = _champion(tmp_path, monkeypatch, d)
        shared = _prompts(champion, _context(d, None))
        mine = _prompts(champion, _context(d, load_profile(d)))
        if d != "banking_knowledge":
            assert mine == shared, d
            continue
        for name, prompt in mine.items():
            assert prompt.startswith(shared[name]), name
            tail = prompt[len(shared[name]) :]
            assert tail.startswith("\n# This dataset's lessons\n")
            assert "`banking_knowledge`'s optimiser" in tail and tail.endswith("\n\n" + guide)


# ── a profile's settings reach the sessions ──────────────────────────────────


def test_a_profile_sets_the_sessions_model_effort_turns_and_what_classic_may_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A classic profile narrowed to `system.md`: the session runs on the profile's model,
    effort and turn limit, the prompt says what is closed, and a write to `helper.py` is
    refused. A model given for one run overrides the profile's."""
    root = _tmp_profiles(tmp_path, monkeypatch)
    _write_profile(
        root,
        "mock",
        profile_yaml="model: sonnet\neffort: high\nmax_turns: 7\nsurfaces:\n  classic: [system.md]\n",
    )
    champion = _champion(tmp_path, monkeypatch, "mock")
    new_dir = tmp_path / "agents" / "mock" / "v4"

    def session(prompt: str) -> None:
        (new_dir / "system.md").write_text("You are a careful agent.\n{policy}\n")
        (new_dir / "diagnosis.json").write_text(json.dumps({"diagnoses": []}))

    sessions = _fake_sdk(monkeypatch, session)
    out = asyncio.run(opt.run_optimiser(champion, "run0", []))
    assert not out.rejected and out.error is None, out.error
    assert out.mode == "classic" and out.routed == ["system.md"]
    options = sessions[0]["options"]
    assert options.model == resolve_model("sonnet") and options.effort == "high"
    assert options.max_turns == 7
    guard = options.hooks["PreToolUse"][0].hooks[0]
    assert not _deny(guard, new_dir / "system.md") and _deny(guard, new_dir / "helper.py")
    assert (
        "This dataset's optimiser profile narrows that to `system.md`: the harness refuses a "
        "write to `helper.py`." in sessions[0]["prompt"]
    )

    asyncio.run(opt.run_optimiser(champion, "run0", [], model="opus"))
    assert sessions[1]["options"].model == resolve_model("opus")


def test_a_routing_profile_drops_a_fix_routed_to_a_closed_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tmp_profiles(tmp_path, monkeypatch)
    _write_profile(
        root,
        "mock",
        profile_yaml="mode: routing\nsurfaces:\n  routing: [system.md, checks.py]\n",
    )
    champion = _champion(tmp_path, monkeypatch, "mock")
    new_dir = tmp_path / "agents" / "mock" / "v4"

    def session(prompt: str) -> None:
        if "**diagnosis** step" in prompt:
            routed = {"checks.py": "origin is not destination", "memory.py": "keep the user"}
            (new_dir / "diagnosis.json").write_text(json.dumps({"surfaces": routed}))
        else:
            (new_dir / "checks.py").write_text(
                "def check_write(name, arguments, state):\n    return None\n"
            )
            (new_dir / "changes.json").write_text(json.dumps({"changes": []}))

    sessions = _fake_sdk(monkeypatch, session)
    out = asyncio.run(opt.run_optimiser(champion, "run0", []))
    assert out.mode == "routing" and out.routed == ["checks.py"]
    assert out.surfaces_changed == ["checks.py"] and not out.rejected, out.error
    assert (
        "lets a diagnosis route only to `system.md`, `checks.py`: `helper.py`, `memory.py`, "
        "`guidance.py` are closed" in sessions[0]["prompt"]
    )
    writer = sessions[1]["options"].hooks["PreToolUse"][0].hooks[0]
    assert _deny(writer, new_dir / "memory.py") and not _deny(writer, new_dir / "checks.py")


# ── a dataset's guards ───────────────────────────────────────────────────────

TODO_GUARD = """def check(files, champion):
    if "TODO" in files.get("system.md", "") and "TODO" not in champion.get("system.md", ""):
        return ["system.md leaves a TODO for the agent"]
    return []
"""


def test_a_dataset_s_guards_refuse_what_its_optimiser_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`optimisers/<domain>/guards.py` runs beside the shared guards: a version it refuses is
    rejected before any challenger runs; one it passes goes through. The guard that runs is the
    one the profile read when the cycle began, whatever is on disk by the end of the session."""
    root = _tmp_profiles(tmp_path, monkeypatch)
    _write_profile(root, "mock", guards_py=TODO_GUARD)
    assert load_profile("mock").guards == TODO_GUARD
    champion = _champion(tmp_path, monkeypatch, "mock")
    prompt_text = {"text": "You are an agent. TODO: tighten this.\n{policy}\n"}

    def session(prompt: str) -> None:  # writes the newest version: v4, then v5
        new_dir = max((tmp_path / "agents" / "mock").iterdir(), key=lambda p: int(p.name[1:]))
        (new_dir / "system.md").write_text(prompt_text["text"])
        (new_dir / "diagnosis.json").write_text(json.dumps({"diagnoses": []}))
        # a session that rewrites its own guard faces the one read when its cycle began
        (root / "mock" / "guards.py").write_text("def check(files, champion):\n    return []\n")

    _fake_sdk(monkeypatch, session)
    out = asyncio.run(opt.run_optimiser(champion, "run0", []))
    assert out.rejected
    assert "guard (mock's optimiser profile): system.md leaves a TODO for the agent" in (
        out.error or ""
    )
    prompt_text["text"] = "You are a careful agent.\n{policy}\n"
    (root / "mock" / "guards.py").write_text(TODO_GUARD)
    out = asyncio.run(opt.run_optimiser(champion, "run0", []))
    assert not out.rejected and out.error is None, out.error


def test_a_guard_that_cannot_run_refuses_rather_than_passes() -> None:
    files, champion = {"system.md": "new"}, {"system.md": "old"}
    assert profile_violations(None, files, champion) == []
    where = "optimisers/mock/guards.py"
    cases: dict[str, tuple[str, list[str] | None]] = {
        "a sentence": (
            "def check(files, champion):\n    return 'one sentence'\n",
            ["one sentence"],
        ),
        "nothing": ("def check(files, champion):\n    return None\n", []),
        "no check": ("X = 1\n", None),
        "raises": ("def check(files, champion):\n    raise KeyError('system.md')\n", None),
        "will not compile": ("def check(files, champion)\n", None),
    }
    for case, (source, expected) in cases.items():
        found = profile_violations(source, files, champion, where)
        if expected is not None:
            assert found == expected, case
        else:
            assert len(found) == 1 and found[0].startswith(where), (case, found)


# ── loading, the fingerprint and the ledger record ───────────────────────────


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        ("modle: opus\n", "unknown keys"),
        ("mode: fancy\n", "mode is one of"),
        ("effort: turbo\n", "effort is one of"),
        ("max_turns: 0\n", "positive whole number"),
        ("diff_budget: true\n", "positive whole number"),
        ("model: ''\n", "model is a model name"),
        ("surfaces:\n  classic: [checks.py]\n", "classic may edit only"),
        ("surfaces:\n  routing: []\n", "non-empty list"),
        ("surfaces:\n  judge: [system.md]\n", "names mode"),
        ("- opus\n", "a mapping"),
    ],
)
def test_a_profile_refuses_what_it_does_not_know(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yaml_text: str, match: str
) -> None:
    root = _tmp_profiles(tmp_path, monkeypatch)
    _write_profile(root, "airline", profile_yaml=yaml_text)
    with pytest.raises(ValueError, match=match):
        load_profile("airline")


def test_the_fingerprint_hashes_the_files_present_in_name_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tmp_profiles(tmp_path, monkeypatch)
    d = _write_profile(root, "airline", profile_yaml="mode: classic\n", guide_md="")
    h = hashlib.sha256()
    for name in ("guide.md", "profile.yaml"):
        h.update(name.encode())
        h.update((d / name).read_bytes())
    first = profiles.fingerprint("airline")
    assert first == h.hexdigest()[:12] == load_profile("airline").fingerprint
    (d / "guide.md").write_text("Research first.\n")
    second = profiles.fingerprint("airline")
    (d / "guards.py").write_text(TODO_GUARD)
    assert len({first, second, profiles.fingerprint("airline")}) == 3


def test_the_ledger_record_names_the_profile_and_what_a_flag_overrode() -> None:
    p = load_profile("airline")
    rec = p.ledger_record("opus", "classic")
    assert rec["domain"] == "airline" and rec["fingerprint"] == p.fingerprint
    assert (rec["model"], rec["mode"], rec["effort"], rec["max_turns"]) == (
        "opus",
        "classic",
        "medium",
        120,
    )
    assert rec["budgets"] == {"transcript": 60_000, "scenario": 60_000, "diff": 90_000}
    assert rec["surfaces"] == ["system.md", "helper.py"] and rec["tuned"] is False
    assert rec["guide_chars"] == 0 and rec["guards"] is False and rec["overrides"] == {}
    # the full id of the profile's model is no override; another model or mode is
    assert p.ledger_record(resolve_model("opus"), "classic")["overrides"] == {}
    routing = p.ledger_record("sonnet", "routing")
    assert routing["overrides"] == {"model": "sonnet", "mode": "routing"}
    assert len(routing["surfaces"]) == 5
    json.dumps(rec)  # it goes in a JSON line as it is


def test_bankings_tuned_profile_routes_and_teaches_its_lessons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """s14: banking's optimiser diagnoses and routes, so checkable rules reach the code surfaces,
    and its guide reaches every prompt under its own heading; nothing else moved."""
    p = load_profile("banking_knowledge")
    assert p.mode == "routing" and not p.is_default() and p.guards is None
    assert (p.model, p.effort, p.max_turns) == ("opus", "medium", 120)
    assert "Search depth is the leaders' edge" in p.guide and "check_reply" in p.guide
    assert re.search(r"task_\d{3}", p.guide) is None  # the guide names no task
    champion = _champion(tmp_path, monkeypatch, "banking_knowledge")
    prompts = _prompts(champion, _context("banking_knowledge", p))
    assert all("# This dataset's lessons" in text for text in prompts.values())
