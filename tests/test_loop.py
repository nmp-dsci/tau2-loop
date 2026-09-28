"""The loop's mechanics, offline: the prompt the optimiser gets, and one cycle with a fake optimiser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tau2_loop.loop.run as loop_run
from tau2_loop.agent.versions import load_version
from tau2_loop.eval.results import TaskResult, read_results
from tau2_loop.eval.runner import RunMeta, list_runs, load_run
from tau2_loop.loop.optimiser import OptimiserOutput, build_prompt, condense_trace, failure_details

MOCK_RUN = next((m.run_id for m in list_runs("mock") if m.summary), None)


@pytest.fixture(autouse=True)
def mlflow_writes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The registry's alias mirror, the prompt registry and the cycle log reach the central
    MLflow whenever it is up; no test writes there. Returns the cycles that would be logged."""
    from tau2_loop.tracking import mlflow_log
    from tau2_loop.tracking import registry as reg

    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(reg, "_mirror_alias", lambda *a, **k: None)
    monkeypatch.setattr(reg, "register_prompt", lambda *a, **k: None)
    monkeypatch.setattr(mlflow_log, "log_cycle", lambda entry: logged.append(dict(entry)))
    return logged


@pytest.mark.skipif(MOCK_RUN is None, reason="no committed mock run")
def test_optimiser_prompt_reads_the_committed_mock_run() -> None:
    meta, results = load_run(str(MOCK_RUN))
    failures = [r for r in results if r.correct is False]
    assert failures, "the committed mock run has failures to diagnose"
    champion = load_version("mock", "v0")
    prompt = build_prompt(champion, "v1", meta.run_id, failures)
    assert "agents/mock/v1/" in prompt and "diagnosis.json" in prompt
    for r in failures:
        assert f"## Task {r.task_id}" in prompt
    assert "### user" in prompt and "### agent" in prompt
    assert "McNemar" in prompt
    trace = Path("runs") / meta.run_id / "traces" / failures[0].trace
    details = failure_details(trace)
    assert "reward 0.0" in details
    assert "termination_reason" in condense_trace(trace)


def _rows(passes: set[str], ids: list[str]) -> list[TaskResult]:
    return [
        TaskResult(t, 1, 1.0 if t in passes else 0.0, t in passes, trace=f"{t}.json") for t in ids
    ]


def test_one_cycle_promotes_and_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake optimiser fixes five tasks and breaks none: promote, test runs compared, ledger complete."""
    import tau2_loop.config as cfg
    import tau2_loop.eval.runner as runner
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / d / "registry.json")
    ids = [str(i) for i in range(20)]
    calls: list[tuple[str, str, str]] = []

    def fake_eval(
        domain: str, agent: str, split: str = "train", **kw: Any
    ) -> tuple[RunMeta, list[TaskResult]]:
        calls.append((domain, agent, split))
        passes = set(ids[:10]) if agent == "v0" else set(ids[:15])
        rows = _rows(passes, ids)
        run_id = f"20260101T000000Z_{domain}_{agent}_{split}_{len(calls)}"
        d = tmp_path / "runs" / run_id
        d.mkdir(parents=True)
        from tau2_loop.eval.results import summarise, write_results

        write_results(d / "results.jsonl", rows)
        meta = RunMeta(
            run_id, domain, agent, "fp" + agent, "m", "u", "j", split, 20, 1, 3, 300, "t", "t"
        )
        meta.summary = summarise(rows).__dict__
        (d / "run.json").write_text(json.dumps(meta.__dict__))
        return meta, rows

    async def fake_optimiser(
        champion: Any, run_id: str, failures: list[TaskResult], model: str = "sonnet"
    ) -> OptimiserOutput:
        assert len(failures) == 10
        return OptimiserOutput(
            "v1",
            {
                "diagnoses": [
                    {"task_id": f.task_id, "surface": "system.md", "change": "x"} for f in failures
                ],
                "prompt_diff_summary": "tightened confirmation rule",
                "helper_diff_summary": "",
                "expected_to_fix": [f.task_id for f in failures],
                "risks": [],
            },
            n_turns=5,
        )

    monkeypatch.setattr(loop_run, "run_eval", fake_eval)
    monkeypatch.setattr(loop_run, "run_optimiser", fake_optimiser)
    monkeypatch.setattr(
        loop_run,
        "load_version",
        lambda d, n: type("V", (), {"domain": d, "name": n, "fingerprint": "fp" + n})(),
    )
    monkeypatch.setattr(
        loop_run,
        "load_run",
        lambda rid: (
            RunMeta(rid, "airline", "v0", "fpv0", "m", "u", "j", "train", 20, 1, 3, 300, "t"),
            _rows(set(ids[:10]), ids),
        ),
    )

    import asyncio

    entry = asyncio.run(loop_run.run_cycle("airline", "v0", "sonnet", 3))
    o = entry["outcome"]
    assert o["verdict"] == "promote" and o["passes"] == "10 → 15"
    assert o["fixed"] == ids[10:15] and o["broken"] == []
    assert o["test_run"] and o["test_passes"] == "15/20"
    # champion, challenger, the challenger's test run, then the champion's (it had none)
    assert [(c[1], c[2]) for c in calls] == [
        ("v0", "train"),
        ("v1", "train"),
        ("v1", "test"),
        ("v0", "test"),
    ]
    tc = o["test_compare"]
    assert tc["passes"] == "10 → 15" and tc["fixed"] == ids[10:15] and tc["broken"] == []
    assert o["pass_1"] == "0.500 → 0.750"
    ledger = led.read_ledger("airline")
    assert len(ledger) == 1 and ledger[0]["outcome"]["verdict"] == "promote"
    assert ledger[0]["diagnoses"][0]["task_id"] == "10"
    r = reg.read_registry("airline")
    assert r["champion"]["agent"] == "v1" and r["champion"]["passed"] == 15
    assert led.next_cycle_number("airline") == 2
    assert read_results(tmp_path / "runs" / o["challenger_run"] / "results.jsonl")[0].task_id == "0"


def test_guard_writes_rejects_a_sibling_version_dir_with_colliding_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """agents/mock/v1 vs agents/mock/v10: a string-prefix check would wrongly let v10 writes through."""
    import asyncio

    import tau2_loop.agent.versions as versions
    import tau2_loop.loop.optimiser as opt

    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    monkeypatch.setattr(opt, "AGENTS_DIR", tmp_path)
    (tmp_path / "mock" / "v0").mkdir(parents=True)
    (tmp_path / "mock" / "v0" / "system.md").write_text("policy")
    (tmp_path / "mock" / "v0" / "agent.yaml").write_text("model: haiku\n")
    champion = versions.load_version("mock", "v0")

    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, options: Any) -> None:
            captured["options"] = options

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self) -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(opt, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(opt, "require_live", lambda: None)
    monkeypatch.setattr(opt, "subscription_env", lambda: {})

    asyncio.run(opt.run_optimiser(champion, "run0", []))

    guard_writes = captured["options"].hooks["PreToolUse"][0].hooks[0]
    new_dir = tmp_path / "mock" / "v1"
    sibling = tmp_path / "mock" / "v10" / "evil.py"
    allowed = new_dir / "system.md"

    denied = asyncio.run(guard_writes({"tool_input": {"file_path": str(sibling)}}, None, None))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    ok = asyncio.run(guard_writes({"tool_input": {"file_path": str(allowed)}}, None, None))
    assert ok == {}


def test_broken_trace_paths_map_each_task_to_its_own_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.eval.results import TaskResult, write_results
    from tau2_loop.loop.optimiser import _broken_trace_paths

    run_dir = tmp_path / "runs" / "20260101T000000Z_x"
    run_dir.mkdir(parents=True)
    write_results(
        run_dir / "results.jsonl",
        [
            TaskResult("3", 1, 0.0, False, trace="3.json"),
            TaskResult("7", 1, 0.0, False, trace="7.json"),
            TaskResult("9", 1, 1.0, True, trace="9.json"),
        ],
    )
    import tau2_loop.loop.optimiser as opt

    monkeypatch.setattr(opt, "RUNS_DIR", tmp_path / "runs")
    paths = _broken_trace_paths({"broken": ["3", "7"], "challenger_run": run_dir.name})
    assert paths == [
        f"runs/{run_dir.name}/traces/3.json",
        f"runs/{run_dir.name}/traces/7.json",
    ]


def test_rejected_optimiser_output_does_not_run_the_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / d / "registry.json")
    ids = [str(i) for i in range(20)]
    evals: list[str] = []

    async def bad_optimiser(*a: Any, **k: Any) -> OptimiserOutput:
        return OptimiserOutput(
            "v1", {}, error="optimiser changed files outside agents/retail/v1: ['src/x.py']"
        )

    monkeypatch.setattr(loop_run, "run_optimiser", bad_optimiser)
    monkeypatch.setattr(loop_run, "run_eval", lambda *a, **k: evals.append("eval"))
    monkeypatch.setattr(
        loop_run,
        "load_version",
        lambda d, n: type("V", (), {"domain": d, "name": n, "fingerprint": "fp"})(),
    )
    monkeypatch.setattr(
        loop_run,
        "_champion_run",
        lambda d, a, c, t=1: (
            RunMeta("r0", d, a, "fp", "m", "u", "j", "train", 20, 1, 3, 300, "t"),
            _rows(set(ids[:10]), ids),
        ),
    )
    import asyncio

    entry = asyncio.run(loop_run.run_cycle("retail", "v0", "sonnet", 3))
    assert entry["outcome"]["verdict"] == "rejected" and evals == []
    assert led.read_ledger("retail")[0]["outcome"]["verdict"] == "rejected"


def test_a_fork_inherits_its_sources_held_challengers_and_names_its_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v3 = v0's files on Sonnet: the optimiser sees v1 and v2 (held against v0) and says Sonnet."""
    import shutil

    import tau2_loop.agent.versions as versions
    import tau2_loop.loop.optimiser as opt
    from tau2_loop.config import AGENTS_DIR, LOOP_DIR
    from tau2_loop.loop import ledger as led

    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    for v in ("v0", "v1", "v2"):
        shutil.copytree(AGENTS_DIR / "airline" / v, tmp_path / "airline" / v)
    shutil.copyfile(LOOP_DIR / "airline" / "ledger.jsonl", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / "ledger.jsonl")
    v3 = versions.fork_version("airline", "v0", model="sonnet")
    assert v3.name == "v3"
    held = opt.held_challengers("airline", "v3")
    assert [h["challenger"] for h in held] == ["v1", "v2"]
    assert all(h["champion"] == "v0" for h in held)
    meta, results = load_run("20260915T132148Z_airline_v2_train")
    failures = [r for r in results if r.correct is False][:1]
    prompt = build_prompt(v3, "v4", meta.run_id, failures)
    assert (
        "The agent is Claude Sonnet 5" in prompt
        and "Claude Haiku 4.5) plays the customer" in prompt
    )
    assert "a fork of `v0`" in prompt and "agents/airline/v2/" in prompt
    assert "breaks none" in prompt


# ── make challenge: a hand-made version through the cycle's own gate ─────────

TRAIN = [f"tr{i}" for i in range(20)]
TEST = [f"te{i}" for i in range(20)]
# (agent, split) → the tasks that pass: v3 fails 6 train tasks, Opus fixes 3 and breaks none
PASSES = {
    ("v3", "train"): set(TRAIN[:14]),
    ("v5", "train"): set(TRAIN[:17]),
    ("v3", "test"): set(TEST[:15]),
    ("v5", "test"): set(TEST[:16]),
}


def _write_run(
    runs: Path, run_id: str, agent: Any, split: str, scored: bool = True, started_at: str = ""
) -> str:
    """A run folder as `run_eval` leaves it; `scored=False` is one still running (or dead)."""
    from dataclasses import asdict

    from tau2_loop.eval.results import summarise, write_results

    ids = TRAIN if split == "train" else TEST
    rows = _rows(PASSES[(agent.name, split)], ids)
    d = runs / run_id
    d.mkdir(parents=True)
    meta = RunMeta(
        run_id=run_id,
        domain="airline",
        agent=agent.name,
        fingerprint=agent.fingerprint,
        model="claude-sdk/" + agent.config.model,
        user_model="u",
        judge_model="j",
        split=split,
        n_tasks=len(ids),
        trials=1,
        concurrency=3,
        seed=300,
        started_at=started_at or "2026-09-28T06:00:00+00:00",
        tau2_sha="t",
        task_ids=ids,
        split_version=2,
    )
    if scored:
        write_results(d / "results.jsonl", rows)
        meta.summary = asdict(summarise(rows))
        meta.finished_at = meta.started_at
    (d / "run.json").write_text(json.dumps(asdict(meta)))
    return run_id


def _challenge_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v5_train: str = "scored"
) -> dict[str, Any]:
    """agents/, runs/ and loop/ under tmp_path, shaped like airline today: v3 (Sonnet) is the
    champion with a train and a test run, v4 a loop challenger, v5 = `fork` of v3 on Opus.
    `v5_train` is v5's train run: "scored", "running" (started now, no summary), "dead"
    (no summary, two days old) or "none". run_eval is faked and records what it was asked."""
    from datetime import UTC, datetime, timedelta

    import tau2_loop.agent.versions as versions
    import tau2_loop.config as cfg
    import tau2_loop.eval.runner as runner
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    runs = tmp_path / "runs"
    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(cfg, "RUNS_DIR", runs)
    monkeypatch.setattr(runner, "RUNS_DIR", runs)
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / "loop" / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / "loop" / d / "registry.json")
    monkeypatch.setattr(loop_run, "split_ids", lambda d, s: {"train": TRAIN, "test": TEST}[s])

    for name, prompt in (
        ("v3", "Be an airline agent.\n{policy}\n"),
        ("v4", "Confirm first.\n{policy}\n"),
    ):
        vdir = tmp_path / "agents" / "airline" / name
        vdir.mkdir(parents=True)
        (vdir / "system.md").write_text(prompt)
        (vdir / "agent.yaml").write_text("model: sonnet\neffort: medium\n")
    v3 = versions.load_version("airline", "v3")
    v5 = versions.fork_version("airline", "v3", model="opus")
    assert v5.name == "v5"

    world: dict[str, Any] = {"calls": []}
    world["v3_train"] = _write_run(runs, "20260928T060029Z_airline_v3_train", v3, "train")
    world["v3_test"] = _write_run(runs, "20260928T073602Z_airline_v3_test", v3, "test")
    reg.promote(world["v3_train"], kind="model swap")
    if v5_train != "none":
        started = datetime.now(UTC) - timedelta(days=2 if v5_train == "dead" else 0)
        world["v5_train"] = _write_run(
            runs,
            "20260928T090000Z_airline_v5_train",
            v5,
            "train",
            scored=v5_train == "scored",
            started_at=started.isoformat(),
        )

    def fake_eval(
        domain: str, agent: str, split: str = "train", **kw: Any
    ) -> tuple[RunMeta, list[TaskResult]]:
        world["calls"].append((agent, split))
        v = versions.load_version(domain, agent)
        run_id = f"20260928T1{len(world['calls']):05d}Z_{domain}_{agent}_{split}"
        _write_run(runs, run_id, v, split)
        return load_run(run_id)

    async def no_optimiser(*a: Any, **k: Any) -> OptimiserOutput:
        raise AssertionError("a challenge never runs the optimiser")

    monkeypatch.setattr(loop_run, "run_eval", fake_eval)
    monkeypatch.setattr(loop_run, "run_optimiser", no_optimiser)
    return world


def test_a_model_swap_challenge_reuses_its_runs_and_records_a_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mlflow_writes: list[dict[str, Any]]
) -> None:
    """v5 = v3's prompt on Opus: v5's train run exists, so only its test split runs; the gate,
    the ledger, the promotion and the test report are the loop's own, with no optimiser."""
    from tau2_loop.llm import model_label
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    world = _challenge_world(tmp_path, monkeypatch)
    entry = loop_run.run_challenge("airline", "v5")

    # (a) the champion's registry run, v5's train run and v3's test run are all reused
    assert world["calls"] == [("v5", "test")]
    o = entry["outcome"]
    assert entry["champion_run"] == world["v3_train"] and o["challenger_run"] == world["v5_train"]
    # (c) the gate's own verdict, in a ledger entry that names the swap and no optimiser
    assert o["verdict"] == "promote" and o["passes"] == "14 → 17"
    assert o["fixed"] == TRAIN[14:17] and o["broken"] == [] and o["still_failed"] == TRAIN[17:]
    [e] = led.read_ledger("airline")
    assert e["cycle"] == 1 and e["kind"] == "model swap" and e["forked_from"] == "v3"
    assert e["champion"] == "v3" and e["challenger"] == "v5"
    assert e["agent_yaml"] == ["model: sonnet → opus"]
    assert e["challenger_model"] == model_label("opus") and e["challenger_effort"] == "medium"
    assert e["optimiser_model"] is None and e["optimiser"] is None
    assert e["diagnoses"] == [] and e["expected_to_fix"] == []
    assert e["failed"] == TRAIN[14:] and e["trials"] == 1 and e["split_version"] == 2
    assert e["outcome"]["verdict"] == "promote" and e["outcome"]["reason"] == o["reason"]
    assert "eval_agent_in" in e["tokens"] and "optimiser_in" not in e["tokens"]
    # promoted as the gate promotes a loop challenger
    r = reg.read_registry("airline")
    assert r["champion"]["agent"] == "v5" and r["history"][-1]["kind"] == "gate"
    # (d) champion vs challenger on test, reported beside the verdict
    tc = e["outcome"]["test_compare"]
    assert tc["champion_run"] == world["v3_test"] and tc["passes"] == "15 → 16"
    assert tc["fixed"] == [TEST[15]] and tc["broken"] == []
    assert e["outcome"]["test_passes"] == "16/20"
    assert mlflow_writes[-1]["kind"] == "model swap"
    # the history the next optimiser session reads says what this cycle was
    history = led.render_history("airline", [])
    assert "v3 → v5, a model swap (model: sonnet → opus), no optimiser · verdict promote" in history


def test_a_challenge_runs_train_only_when_no_run_of_those_bytes_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A train run still in flight is waited for, never duplicated; a dead one does not block."""
    from tau2_loop.loop import ledger as led

    world = _challenge_world(tmp_path, monkeypatch, v5_train="running")
    with pytest.raises(RuntimeError, match="no summary yet"):
        loop_run.run_challenge("airline", "v5")
    assert world["calls"] == [] and led.read_ledger("airline") == []

    world = _challenge_world(tmp_path / "dead", monkeypatch, v5_train="dead")
    entry = loop_run.run_challenge("airline", "v5", run_test=False)
    assert world["calls"] == [("v5", "train")]
    assert entry["outcome"]["verdict"] == "promote" and "test_run" not in entry["outcome"]


def test_a_challenge_refuses_the_champion_and_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from tau2_loop.loop import ledger as led

    world = _challenge_world(tmp_path, monkeypatch)
    agents = tmp_path / "agents" / "airline"
    shutil.copytree(agents / "v3", agents / "v6")  # the champion's bytes under another name
    with pytest.raises(ValueError, match="champion"):
        loop_run.run_challenge("airline", "v3")
    with pytest.raises(ValueError, match="bytes"):
        loop_run.run_challenge("airline", "v6")
    assert world["calls"] == [] and led.read_ledger("airline") == []
