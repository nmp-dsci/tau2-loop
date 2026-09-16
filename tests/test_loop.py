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
    """A fake optimiser fixes five tasks and breaks none: promote, test run recorded, ledger complete."""
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
    assert [c[2] for c in calls] == [
        "train",
        "train",
        "test",
    ]  # champion, challenger, then test once
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
        lambda d, a, c: (
            RunMeta("r0", d, a, "fp", "m", "u", "j", "train", 20, 1, 3, 300, "t"),
            _rows(set(ids[:10]), ids),
        ),
    )
    import asyncio

    entry = asyncio.run(loop_run.run_cycle("retail", "v0", "sonnet", 3))
    assert entry["outcome"]["verdict"] == "rejected" and evals == []
    assert led.read_ledger("retail")[0]["outcome"]["verdict"] == "rejected"
